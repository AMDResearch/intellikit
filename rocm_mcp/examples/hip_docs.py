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
    """Example of using the HIP documentation MCP tool via an LLM agent."""
    # initialize MCP client and tools
    client = MultiServerMCPClient(
        {
            "hip_docs": {
                "transport": "stdio",
                "command": "hip-docs-mcp",
                "args": [],
            }
        }
    )
    tools = await client.get_tools()

    # create LLM model on LLM gateway
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
                    content=("Search for HIP API documentation about memory sharing functions.")
                )
            ]
        },
    )
    print("HIP Documentation response:", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
