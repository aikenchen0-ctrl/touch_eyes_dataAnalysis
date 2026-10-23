"""读取跨端任务标签与复核旁车；不改写原始信号或将场景作为模型输入。"""
import hashlib
import json
from pathlib import Path
from uuid import UUID

REVIEW_CONTRACT = "touch_sample_review_v1"
REVIEW_STATUSES = {"confirmed", "mistake", "uncertain"}
CONTEXT_CONTRACTS = {
    1: "touch_collection_context_v1",
    2: "touch_collection_context_v2",
}
AUDIT_FIELDS = ("taskId", "bodyPosture", "requestedDevicePose", "grip", "support",
                "collectionProtocol", "scheduleId", "cycleIndex", "stepIndex",
                "strengthTruthSource", "labelStatus", "reviewSha256")


def is_raw_sample_path(path):
    return "annotations" not in Path(path).parts


def read_annotation(path, data):
    path = Path(path)
    if not isinstance(data, dict):
        raise RuntimeError("invalid_sample_record")
    context = data.get("collectionContext") or {}
    if not isinstance(context, dict):
        raise RuntimeError("invalid_collection_context")
    if context:
        version = context.get("schemaVersion")
        if version not in CONTEXT_CONTRACTS or context.get("contract") != CONTEXT_CONTRACTS[version]:
            raise RuntimeError("unsupported_collection_context")
        if version == 2:
            validate_interleaved_context(context)
    result = {key: str(context.get(key) or "unknown") for key in AUDIT_FIELDS}
    result["labelStatus"] = str(context.get("labelStatus") or "legacy_unreviewed")
    result["reviewSha256"] = ""
    if result["labelStatus"] not in {"task_assigned", "legacy_unreviewed"}:
        raise RuntimeError("invalid_collection_label_status")
    sample_id = str(data.get("id") or "")
    if not sample_id and not context:
        # 兼容只有旧特征的诊断记录；新任务样本必须有可复核身份。
        return result
    # 旧记录可以没有 UUID；只允许安全文件名，禁止经样本标识越出目录。
    if not sample_id or Path(sample_id).name != sample_id or any(c in sample_id for c in "/\\:"):
        raise RuntimeError("invalid_sample_identity")
    sidecar = path.parent / "annotations" / (sample_id + ".json")
    if Path(str(sidecar) + ".bak").exists() or Path(str(sidecar) + ".new").exists():
        raise RuntimeError("sample_review_transaction_incomplete")
    if not sidecar.exists():
        return result
    try:
        UUID(sample_id)
        raw = sidecar.read_bytes()
        review = json.loads(raw)
        if (review["schemaVersion"] != 1 or review["contract"] != REVIEW_CONTRACT
                or review["sampleId"] != sample_id
                or review["sampleFileName"] != path.name):
            raise ValueError("identity")
        history = review["history"]
        if not isinstance(history, list) or not history:
            raise ValueError("history")
        for revision, entry in enumerate(history, 1):
            if (entry["revision"] != revision or entry["status"] not in REVIEW_STATUSES
                    or entry["source"] != "user_review"
                    or not isinstance(entry["timestampMs"], (int, float))
                    or not 0 < entry["timestampMs"] < 2**63):
                raise ValueError("revision")
        if review["labelStatus"] != history[-1]["status"]:
            raise ValueError("status")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("invalid_sample_review") from error
    result["labelStatus"] = review["labelStatus"]
    result["reviewSha256"] = hashlib.sha256(raw).hexdigest()
    return result


def validate_interleaved_context(context):
    """交错协议必须能证明每轮四类各一次，且不依赖场景推断轻重。"""
    actions = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
    schedule_id = context.get("scheduleId")
    cycle = context.get("cycleIndex")
    step = context.get("stepIndex")
    order = context.get("scheduleOrder")
    if (context.get("collectionProtocol") != "interleaved_v2"
            or context.get("strengthTruthSource") != "user_task"
            or not isinstance(schedule_id, str) or not schedule_id
            or not isinstance(cycle, int) or isinstance(cycle, bool) or cycle < 0
            or not isinstance(step, int) or isinstance(step, bool) or not 0 <= step < 4
            or not isinstance(order, list) or sorted(order) != sorted(actions)):
        raise RuntimeError("invalid_interleaved_context")


def conflicting_duplicate_reviews(paths):
    """相同原始内容的不同副本若复核不同，全部隔离，避免用未复核副本绕过排除。"""
    decisions = {}
    conflicts = set()
    for path in paths:
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            data = json.loads(raw)
            decision = read_annotation(path, data)["labelStatus"]
        except (ValueError, RuntimeError):
            decision = "invalid"
        if digest in decisions and decisions[digest] != decision:
            conflicts.add(digest)
        decisions[digest] = decision
    return conflicts


def exclusion_reason(annotation):
    status = annotation["labelStatus"]
    return "user_review_" + status if status in {"mistake", "uncertain"} else ""


def require_trainable(path, data):
    annotation = read_annotation(path, data)
    reason = exclusion_reason(annotation)
    if reason:
        raise RuntimeError(reason)
    return annotation
