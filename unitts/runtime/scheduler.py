from __future__ import annotations

from dataclasses import dataclass

from unitts.contracts import AdmissionError, ExecutionPlan, SynthesisRequest
from unitts.runtime.registry import EngineRegistry
from unitts.runtime.strategies import STRATEGIES


@dataclass(frozen=True)
class AdmissionPolicy:
    max_active_per_tenant: int = 4
    queue_timeout_seconds: float = 30.0


class GlobalAdmissionScheduler:
    """Owns quotas and replica selection, not model-specific batch formation."""

    def __init__(self, registry: EngineRegistry, policy: AdmissionPolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or AdmissionPolicy()
        self._active: dict[str, int] = {}

    def plan(self, request: SynthesisRequest, *, tenant_id: str = "local", priority: int = 0) -> ExecutionPlan:
        active = self._active.get(tenant_id, 0)
        if active >= self.policy.max_active_per_tenant:
            raise AdmissionError("Tenant concurrency quota exceeded", details={"tenant_id": tenant_id})
        manifest = self.registry.get(request.engine_id or "dummy")
        strategy_name = manifest.capabilities.execution_strategy
        estimate = STRATEGIES[strategy_name]().estimate(request)
        self._active[tenant_id] = active + 1
        return ExecutionPlan(
            request_id=request.request_id, engine_id=manifest.engine_id,
            replica_id=f"{manifest.engine_id}-local-0", strategy=strategy_name,
            tenant_id=tenant_id, priority=priority, resource_estimate=estimate,
        )

    def complete(self, plan: ExecutionPlan) -> None:
        self._active[plan.tenant_id] = max(0, self._active.get(plan.tenant_id, 1) - 1)
