import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

from test_agents import _plan

from elenchix.agents.assessment import AssessmentAgent
from elenchix.agents.teaching import TeachingAgent
from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.llm import OpenAICompatibleLLM
from elenchix.schemas import DialogueMessage


def setup(role, responses, *, streaming=False):
    graph = JsonGraphStore(load_config(Path(__file__).parent / "fixtures/config.yaml").graph)
    requests = []
    outputs = iter(responses)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(outputs)

    client = NS(chat=NS(completions=NS(create=create)))
    client.with_options = lambda **kwargs: client
    llm = OpenAICompatibleLLM.__new__(OpenAICompatibleLLM)
    llm.roles = {role: NS(model="test-model", temperature=0, extra_body={"enable_thinking": True} if streaming else {})}
    llm.clients = {role: client}
    llm.assessment_max_retries = 1
    llm.teaching_max_empty_retries = 1
    llm.teaching_history_limit = 10
    return graph, llm, requests


def response(content="", calls=()):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


def test_teacher_does_not_load_graph_when_tool_is_not_requested(monkeypatch):
    graph, llm, requests = setup("teaching", [response("What evidence would you gather?")])

    def unexpected(*args):
        raise AssertionError("graph context must be loaded only on demand")

    monkeypatch.setattr(graph, "case_subgraph", unexpected)
    turn = TeachingAgent(graph, llm).teach(_plan())
    assert turn.tutor_message == "What evidence would you gather?"
    assert len(requests) == 1
    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["tools"][0]["function"]["name"] == "get_case_subgraph"
    assert "PREREQUISITE_FOR" not in json.dumps(requests[0]["messages"])


def test_assessor_gets_case_graph_only_in_response_to_its_tool_call():
    call = NS(id="graph_call", function=NS(name="get_case_subgraph", arguments="{}"))
    result = json.dumps({"reasoning": "No assessable target evidence.", "abilities": {}, "entities": {}})
    graph, llm, requests = setup("assessment", [response(calls=[call]), response(result)])
    assessment = AssessmentAgent(graph, graph.config, llm).assess_dialogue(_plan(), [
        DialogueMessage(role="teacher", content="What information is needed?"),
        DialogueMessage(role="learner", content="Please explain the task."),
    ])
    assert assessment.events == []
    assert len(requests) == 2
    assert "PREREQUISITE_FOR" not in requests[0]["messages"][0]["content"]
    tool_result = requests[1]["messages"][-1]
    assert tool_result["role"] == "tool" and tool_result["tool_call_id"] == "graph_call"
    context = json.loads(tool_result["content"])
    assert context == graph.case_subgraph("case_demo_01")
    assert "kp_demo_differential" not in context["knowledge_ids"]


def test_streamed_teacher_can_request_graph_then_return_visible_text():
    def chunk(content=None, calls=()):
        return NS(choices=[NS(delta=NS(content=content, tool_calls=calls, reasoning_content="hidden"))])

    first = [
        chunk(calls=[NS(index=0, id="call_1", function=NS(name="get_case_", arguments="{"))]),
        chunk(calls=[NS(index=0, id=None, function=NS(name="subgraph", arguments="}"))]),
    ]
    final = [chunk("Explain "), chunk("your reasoning.")]
    graph, llm, requests = setup("teaching", [iter(first), iter(final)], streaming=True)
    turn = TeachingAgent(graph, llm).teach(_plan())
    assert turn.tutor_message == "Explain your reasoning."
    assert len(requests) == 2 and all(item["stream"] for item in requests)
    assert "PREREQUISITE_FOR" not in json.dumps(requests[0]["messages"])
    assert requests[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert json.loads(requests[1]["messages"][-1]["content"])["case_id"] == "case_demo_01"
