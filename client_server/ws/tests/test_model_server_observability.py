import asyncio
import logging

from client_server.ws.model_server import PolicyServer, _action_steps
from client_server.ws.protocol.messages import MessageType
from client_server.ws.protocol.schemas import Frame


def test_action_steps_reports_common_policy_reply_shapes():
    assert _action_steps([{"action": 1}, {"action": 2}]) == 2
    assert _action_steps({"actions": [1, 2, 3], "latency_ms": 1.0}) == 3
    assert _action_steps(object()) is None


def test_infer_logs_start_completion_latency_and_horizon(caplog):
    class Model:
        def update_obs(self, observation):
            self.observation = observation

        def get_action(self):
            return [{"action": 1}, {"action": 2}]

    frame = Frame(
        message_type=MessageType.INFER,
        request_id="request-1",
        evaluation_id="evaluation-1",
        trial_id="trial-1",
        step=4,
        payload={"observation": {"state": [0.0]}},
    )
    caplog.set_level(logging.INFO, logger="client_server.ws.model_server")
    response = asyncio.run(PolicyServer(Model()).process_frame(frame))
    assert response.message_type == MessageType.INFER_RESULT
    assert "[INFER] start request=request-1 trial=trial-1 step=4" in caplog.text
    assert "[INFER] done request=request-1" in caplog.text
    assert "action_steps=2" in caplog.text
