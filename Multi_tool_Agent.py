from langchain_core.prompts import MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_experimental.utilities import PythonREPL

from langchain.prompts import ChatPromptTemplate , PromptTemplate,MessagesPlaceholder
from langchain_core.messages import (

    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ChatMessage
)

from langgraph.prebuilt import ToolExecutor , ToolInvocation
from langgraph.graph import END , StateGraph

from typing import Annotated , Tuple , Sequence , List ,TypedDict
import json
import operator
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser , StrOutputParser

from langchain.tools import tool



search_tool = TavilySearchResults(max_results=1)
repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    if any error occurs while running the code go back to the agent or code provider to see the error and give the new code and run again.


    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"


@tool
def Blog_writer(data: Annotated[str , "It contains the blog data"],
                filename : str = "blog_file.txt" ):
    """  Use this  data to write a blog and save it in a file based on the user prefered filename if user doesn't give a filename 

        use default filename
    """
    with open(filename,'w') as f:
        f.write(data)
    f.close()
    return "blog file is saved "


@tool
def LinkedinPost(data : Annotated[str , "It contains the linkedin post data with emoji's"],

                filename : str = "linked_file.txt" ):

    """  Use this  data to write a linkedin post and save it in a file based on the user prefered filename if user doesn't give a filename 

        use default filename
    """
    with open(filename,'w') as f:
        f.write(data)
    f.close()
    return "linkedin post file saved is saved "




tools = [search_tool , python_repl , Blog_writer ,LinkedinPost]

tool_execute = ToolExecutor(tools)


from langchain.tools.render import format_tool_to_openai_function
tool_func = [format_tool_to_openai_function(tool) for tool in tools]

llm = ChatOpenAI(model='gpt-3.5-turbo-1106')
llm_func = llm.bind_functions(tool_func)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm_func.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = tool_execute.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)


# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
de = app.invoke(inputs)

#print(de)
#for output in app.stream(inputs):
    # stream() yields dictionaries with output keyed by node name
#    for key, value in output.items():
#        print(f"Output from node '{key}':")
#        print("---")
#        print(value)
#    print("\n---\n")


inputs = {"messages": [HumanMessage(content="Take a input list = [1,53,23,2,34,14,88] use quick sort to sort the list .  ")]}

for s in app.stream(
    inputs,
    
    {"recursion_limit": 150},
):
    print(s)
    print("---")


inputs = {"messages": [HumanMessage(content="write a blog on LLM's and save the file in blog1.txt")]}
de = app.invoke(inputs)

#print(de)
