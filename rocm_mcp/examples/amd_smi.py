# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import getpass
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def main() -> None:
    """Example of using the amd-smi MCP tools via an LLM agent.

    The amd-smi MCP server exposes GPU system management tools:
      - get_driver_information: Driver version, name, and date
      - list_gpus: GPU index, BDF address, and UUID for each GPU
      - get_gpu_static_info: Hardware specs (VRAM, compute units, power caps)
      - get_gpu_metrics: Real-time activity, temperature, power, clocks, VRAM usage
      - get_gpu_firmware_info: VBIOS and firmware block versions
      - get_gpu_process_info: Processes using each GPU
      - get_gpu_bad_pages: ECC error counts and retired memory pages
    """
    client = MultiServerMCPClient(
        {
            "amd-smi": {
                "transport": "stdio",
                "command": "amd-smi-mcp",
                "args": [],
            }
        }
    )
    tools = await client.get_tools()

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
                        "Give me a health summary of the GPUs in this system. Include the driver"
                        " version, GPU model names, current temperatures and power draw, VRAM"
                        " usage, and whether there are any ECC errors or bad pages."
                    )
                )
            ]
        },
    )
    print("AMD SMI response:", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
