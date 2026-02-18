#!/usr/bin/env python3
"""
binbang CrewAI 진입점

실행 방법:
    crewai run                                        # 대화형으로 topic 입력
    python main.py --topic "LoginPromptModal 구현"    # 직접 실행
    CREW_TOPIC="LoginPromptModal 구현" crewai run     # 환경변수로 topic 전달

환경변수 (.env):
    GEMINI_API_KEY  — Gemini API 키 (필수)
    MODEL           — 사용할 모델 (기본: gemini/gemini-2.5-flash)
    REPO_ROOT       — 파일을 저장할 모노레포 루트 경로 (필수, 예: /home/ubuntu/workspace/binbang)
    CREW_TOPIC      — crewai run 시 topic 자동 입력 (선택)
    CREW_VERBOSE    — 상세 로그 출력 (기본: true)
    CREW_DRY_RUN    — true 시 파일 저장 없이 경로만 출력 (기본: false)
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


_CODE_BLOCK_RE = re.compile(
    r"```(?:typescript|tsx|ts|javascript|js)\n(// (?:apps|packages)/[^\n]+\n.*?)```",
    re.DOTALL,
)


def _extract_files(task_raw: str) -> dict[str, str]:
    """코드블록 첫 줄의 '// 경로' 주석을 파싱해 {상대경로: 내용} 딕셔너리를 반환한다."""
    files: dict[str, str] = {}
    for match in _CODE_BLOCK_RE.finditer(task_raw):
        block = match.group(1)
        lines = block.split("\n")
        rel_path = lines[0].strip()[3:]
        content = "\n".join(lines[1:]).rstrip("\n") + "\n"
        files[rel_path] = content
    return files


def _apply_files(files: dict[str, str], repo_root: str) -> None:
    """추출된 파일을 repo_root 기준으로 저장한다."""
    if not files:
        print("  저장할 파일이 없습니다.")
        return
    dry_run = os.environ.get("CREW_DRY_RUN", "false").lower() == "true"
    for rel_path, content in sorted(files.items()):
        abs_path = Path(repo_root) / rel_path
        if dry_run:
            print(f"  [DRY-RUN] {rel_path}")
            continue
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        print(f"  저장: {rel_path}")


def _validate_env() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  .env 파일에 GEMINI_API_KEY=... 를 추가하세요.")
        sys.exit(1)


def _resolve_topic(topic: str | None) -> str:
    """topic을 결정한다. 우선순위: 인자 > 환경변수 > 대화형 입력"""
    if topic:
        return topic
    env_topic = os.environ.get("CREW_TOPIC", "").strip()
    if env_topic:
        return env_topic
    # crewai run 처럼 인자 없이 호출된 경우 대화형 입력
    print("구현할 기능 또는 태스크를 입력하세요.")
    print("예시: LoginPromptModal 컴포넌트 구현")
    print("예시: travel-guest-cleanup worker job 수정")
    topic = input("\nTopic > ").strip()
    if not topic:
        print("[ERROR] topic이 입력되지 않았습니다.")
        sys.exit(1)
    return topic


def run(topic: str | None = None) -> None:
    """crewai run 또는 직접 호출 시 진입점"""
    from binbang_crew.crew import BinbangCrew

    _validate_env()
    topic = _resolve_topic(topic)
    default_model = os.environ.get("MODEL", "gemini/gemini-2.5-flash-preview-04-17")
    fallback_models = [
        m.strip()
        for m in os.environ.get("MODEL_FALLBACKS", "").split(",")
        if m.strip()
    ]
    max_retries = int(os.environ.get("CREW_RUN_RETRIES", "3"))
    retry_delay_sec = float(os.environ.get("CREW_RETRY_DELAY_SEC", "2"))

    models_to_try: list[str] = []
    for model in [default_model, *fallback_models]:
        if model not in models_to_try:
            models_to_try.append(model)

    print(f"\n{'='*60}")
    print("  binbang CrewAI 시작")
    print(f"  Topic : {topic}")
    print(f"  Model : {default_model}")
    if fallback_models:
        print(f"  Fallback Models : {', '.join(fallback_models)}")
    print(f"  Retry : model당 최대 {max_retries}회")
    print(f"{'='*60}\n")

    last_error: Exception | None = None
    for model in models_to_try:
        os.environ["MODEL"] = model
        for attempt in range(1, max_retries + 1):
            try:
                result = BinbangCrew().crew().kickoff(inputs={"topic": topic})
                print(f"\n{'='*60}")
                print("  최종 결과")
                print(f"{'='*60}")
                print(result)
                print("\n로그 파일: crew_output.log")

                final_raw = result.raw if hasattr(result, "raw") else str(result)
                if "REWORK" in final_raw.upper():
                    print("\n[SKIP] final_review 결과가 REWORK — 파일을 저장하지 않습니다.")
                    print("       crew_output.log에서 수정 지시를 확인하세요.")
                    return

                repo_root = os.environ.get("REPO_ROOT", "").strip()
                if not repo_root:
                    print("\n[SKIP] REPO_ROOT 환경변수가 없어 파일을 저장할 수 없습니다.")
                    print("       .env에 REPO_ROOT=/path/to/binbang 을 추가하세요.")
                    return

                all_files: dict[str, str] = {}
                for task_out in getattr(result, "tasks_output", []):
                    if getattr(task_out, "name", None) in ("develop", "develop_worker"):
                        all_files.update(_extract_files(getattr(task_out, "raw", "")))

                print(f"\n{'='*60}")
                print(f"  파일 저장 ({len(all_files)}개) — REPO_ROOT: {repo_root}")
                print(f"{'='*60}")
                _apply_files(all_files, repo_root)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc)
                is_retryable = (
                    "503" in message
                    or "UNAVAILABLE" in message
                    or "None or empty" in message
                    or "timeout" in message.lower()
                )
                is_last_attempt = attempt >= max_retries
                if not is_retryable or is_last_attempt:
                    print(
                        f"[WARN] 모델={model}, 시도={attempt}/{max_retries} 실패: {message}"
                    )
                    break
                wait_sec = retry_delay_sec * (2 ** (attempt - 1))
                print(
                    "[WARN] 일시적 LLM 오류. "
                    f"모델={model}, 시도={attempt}/{max_retries}, "
                    f"{wait_sec:.1f}초 후 재시도"
                )
                time.sleep(wait_sec)

    print("[ERROR] Crew 실행에 실패했습니다.")
    if last_error is not None:
        print(f"원인: {last_error}")
    sys.exit(1)


def train() -> None:
    """crewai train — 반복 실행으로 결과 개선"""
    from binbang_crew.crew import BinbangCrew

    _validate_env()
    topic = _resolve_topic(None)
    n_iterations = int(os.environ.get("CREW_TRAIN_ITERATIONS", "2"))

    print(f"학습 모드: {n_iterations}회 반복, topic={topic}")
    BinbangCrew().crew().train(
        n_iterations=n_iterations,
        filename="crew_training.pkl",
        inputs={"topic": topic},
    )


def replay() -> None:
    """crewai replay — 특정 task_id부터 재실행"""
    from binbang_crew.crew import BinbangCrew

    task_id = os.environ.get("CREW_REPLAY_TASK_ID", "").strip()
    if not task_id:
        task_id = input("재실행할 task_id > ").strip()
    if not task_id:
        print("[ERROR] task_id가 필요합니다.")
        sys.exit(1)

    print(f"task_id={task_id} 부터 재실행합니다.")
    BinbangCrew().crew().replay(task_id=task_id)


def test() -> None:
    """crewai test — 결과 평가 (n회 실행 후 점수 측정)"""
    from binbang_crew.crew import BinbangCrew

    _validate_env()
    topic = _resolve_topic(None)
    n_iterations = int(os.environ.get("CREW_TEST_ITERATIONS", "1"))
    eval_model = os.environ.get(
        "CREW_EVAL_MODEL",
        os.environ.get("MODEL", "gemini/gemini-2.5-flash-preview-04-17"),
    )

    print(f"테스트 모드: {n_iterations}회, eval_model={eval_model}, topic={topic}")
    BinbangCrew().crew().test(
        n_iterations=n_iterations,
        eval_llm=eval_model,
        inputs={"topic": topic},
    )


def run_with_trigger() -> None:
    """외부 트리거(webhook 등)로 호출 시 진입점 — topic을 환경변수로 전달"""
    topic = os.environ.get("CREW_TOPIC", "").strip()
    if not topic:
        print("[ERROR] CREW_TOPIC 환경변수가 필요합니다.")
        print("  CREW_TOPIC='LoginPromptModal 구현' crewai run_with_trigger")
        sys.exit(1)
    run(topic=topic)


# ──────────────────────────────────────────────
# 직접 실행 시 (python main.py --topic "...")
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="binbang CrewAI — 멀티 에이전트 개발 워크플로우"
    )
    parser.add_argument(
        "--topic",
        required=False,
        help="구현할 기능 설명 (생략 시 대화형 입력)",
    )
    parser.add_argument(
        "--train", action="store_true", help="학습 모드로 실행"
    )
    parser.add_argument(
        "--replay", action="store_true", help="특정 task부터 재실행"
    )
    parser.add_argument(
        "--test", action="store_true", help="테스트 모드로 실행"
    )
    args = parser.parse_args()

    if args.train:
        train()
    elif args.replay:
        replay()
    elif args.test:
        test()
    else:
        run(topic=args.topic)
