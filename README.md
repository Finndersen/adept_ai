# Your Project Name

A brief description of what your project does and who it's for.


## Features

### General
- Fully async


### MCP Capabilities
- Add description, instructions and usage examples for how to use the MCP server
- Choose which tools to use
- Automatically include resources in system prompt context data
- Handle server sampling requests (so MCP server can make request to LLM/agent)
- Handle tool and resource list changed notifications to reset caches (not even officially supported by the MCP SDK yet)

## Installation

Instructions on how to install and set up your project.

## Usage

Examples of how to use your project.

### Customise MCP Capabilities

You can create a custom `StdioMCPCapability` or `HttpMCPCapability` subclass to do things like:
- Customize the behaviour of specific tools, by defining a tool that wraps one or more MCP server tools
- Create a hybrid capability which has a mix of MCP and non-MCP tools (and resources etc)
- 


## Contributing

Guidelines for contributing to your project.

## TODO

### General
- More logging / observability 

### MCP Capability
- Smarter caching for tools list and resources
- Subscribe to resources for automatic dynamic updates and smart caching
- Dynamic / Templated resources
- MCP prompts


## License

Information about the project's license.
