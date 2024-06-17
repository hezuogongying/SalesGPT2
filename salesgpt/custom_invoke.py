# 更正进口声明
import inspect
from typing import Any, Dict, Optional

# 更正了 RunnableConfig 的导入路径
from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.chains.base import Chain
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables import RunnableConfig, ensure_config


class CustomAgentExecutor(AgentExecutor):
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        """
        使用给定的输入和配置调用代理。

        Args:
            input: 要传递给代理的输入数据。
            config: 代理的配置.
            **kwargs: 要传递给代理的其他关键字参数。

        Returns:Dict[str, Any]
            代理的输出.

        """
        intermediate_steps = []  # 初始化列表以捕获中间步骤

        # 确保配置设置正确
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name")
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        # 根据提供的输入准备输入
        inputs = self.prep_inputs(input)
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        # 检查 _call 方法是否支持新参数 'run_manager'
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            name=run_name,
        )

        # 捕获链的开头作为中间步骤
        intermediate_steps.append(
            {"event": "Chain Started", "details": "Inputs prepared"}
        )

        try:
            # 执行 _call 方法，如果支持则传递 'run_manager'
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )
            # 捕获成功的调用作为中间步骤
            intermediate_steps.append({"event": "Call Successful", "outputs": outputs})
        except BaseException as e:
            # 处理错误并将其捕获为中间步骤
            run_manager.on_chain_error(e)
            intermediate_steps.append({"event": "Error", "error": str(e)})
            raise e
        finally:
            # 标记链执行结束
            run_manager.on_chain_end(outputs)

        # 准备最终输出，包括运行信息（如果需要）
        final_outputs: Dict[str, Any] = self.prep_outputs(
            inputs, outputs, return_only_outputs
        )
        if include_run_info:
            final_outputs["run_info"] = RunInfo(run_id=run_manager.run_id)

        # 在最终输出中包含中间步骤
        final_outputs["intermediate_steps"] = intermediate_steps

        return final_outputs


if __name__ == "__main__":
    agent = CustomAgentExecutor()
