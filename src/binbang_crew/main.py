#!/usr/bin/env python3
"""
binbang CrewAI 진입점

실행 방법:
    crewai run                                        # 대화형으로 topic 입력
    python main.py --topic "LoginPromptModal 구현"    # 직접 실행
    CREW_TOPIC="LoginPromptModal 구현" crewai run     # 환경변수로 topic 전달

환경변수 (.env):
    GEMINI_API_KEY  — Gemini API 키 (필수)
    MODEL           — 사용할 모델 (기본: gemini/gemini-2.5-flash-preview-04-17)
    CREW_TOPIC      — crewai run 시 topic 자동 입력 (선택)
    CREW_VERBOSE    — 상세 로그 출력 (기본: true)
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


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
    model = os.environ.get("MODEL", "gemini/gemini-2.5-flash-preview-04-17")

    print(f"\n{'='*60}")
    print("  binbang CrewAI 시작")
    print(f"  Topic : {topic}")
    print(f"  Model : {model}")
    print(f"{'='*60}\n")

    result = BinbangCrew().crew().kickoff(inputs={"topic": topic})

    print(f"\n{'='*60}")
    print("  최종 결과")
    print(f"{'='*60}")
    print(result)
    print("\n로그 파일: crew_output.log")


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

