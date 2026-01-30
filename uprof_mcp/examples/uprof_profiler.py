# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

import argparse
import asyncio
import getpass
import os
import tempfile

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from uprof_mcp.uprof_profiler import UProfProfiler

load_dotenv()


async def agentic_hotspot_analysis(executable: str, executable_args: list[str]) -> None:
    """Profile an executable using an LLM agent and UProf Profiler MCP tool."""
    print("Running in agentic mode...")

    # initialize MCP client and tools
    client = MultiServerMCPClient(
        {
            "profile": {
                "transport": "stdio",
                "command": "uprof-profiler-mcp",
                "args": [],
            }
        }
    )
    tools = await client.get_tools()

    # create LLM model to support LLM gateway
    model = ChatOpenAI(
        model="gpt-5-mini",
        max_retries=2,
        api_key="dummy",
        base_url="https://llm-api.amd.com/OpenAI",
        default_headers={
            "Ocp-Apim-Subscription-Key": os.environ.get("LLM_GATEWAY_KEY"),
            "user": getpass.getuser(),
        },
        temperature=0,
    )
    agent = create_agent(model, tools)

    response = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Profile the executable in {executable} with arguments "
                        f"[{executable_args}] to identify and report the top 5 functions that "
                        "consume the most CPU time."
                    )
                )
            ]
        },
    )
    print("Profiling response:", response["messages"][-1].content)


def non_agentic_hotspot_analysis(executable: str, executable_args: list[str]) -> None:
    """Profile an executable using UProf Profiler MCP tool in non-agentic mode."""
    print("Running in non-agentic mode...")

    profiler = UProfProfiler()
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = profiler.find_hotspots(
            output_dir=tmpdirname,
            executable=executable,
            executable_args=executable_args,
        )
        with result.report_path.open() as report_file:
            report_content = report_file.read()
            print("Profiling report:", report_content)


async def main() -> None:
    """Example of using UProf Profiler MCP with LangChain Agent or non-agentic mode."""
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Example of using UProf Profiler MCP with LangChain Agent"
    )
    parser.add_argument(
        "--executable", type=str, required=True, help="Path to the executable to profile"
    )
    parser.add_argument(
        "--args",
        type=str,
        nargs="*",
        default=[],
        help="Arguments to pass to the executable",
    )
    parser.add_argument("--classic", action="store_true", help="Run in non-agentic mode")
    args = parser.parse_args()
    executable = args.executable
    executable_args = args.args
    print(f"Executable: {executable} Arguments: {executable_args}")

    # profile using agentic or non-agentic mode
    if not args.classic:
        await agentic_hotspot_analysis(executable, executable_args)
    else:
        non_agentic_hotspot_analysis(executable, executable_args)


if __name__ == "__main__":
    asyncio.run(main())
