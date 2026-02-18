#!/usr/bin/env python3
"""
binbang CrewAI — 멀티 에이전트 개발 워크플로우

실행:
    crewai run  (main.py 경유)
    python -m binbang_crew.main --topic "LoginPromptModal 컴포넌트 구현"

환경변수 (.env):
    ANTHROPIC_API_KEY   — Claude API 키 (필수)
    CREW_MODEL          — 사용할 모델 (기본: claude-opus-4-6)
    CREW_VERBOSE        — 상세 로그 출력 (기본: true)
"""

import os
import sys

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class BinbangCrew:
    """binbang 모노레포 개발 자동화 Crew"""

    # @CrewBase 가 이 파일 위치 기준으로 config/ 를 자동으로 찾음
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def _llm(self) -> LLM:
        model = os.environ.get("CREW_MODEL", "claude-opus-4-6")
        return LLM(
            model=f"anthropic/{model}",
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.2,  # 일관된 코드/설계 출력을 위해 낮게 설정
        )

    # ──────────────────────────────────────────────
    # Agents
    # ──────────────────────────────────────────────

    @agent
    def tech_lead(self) -> Agent:
        return Agent(
            config=self.agents_config["tech_lead"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=3,
        )

    @agent
    def ux_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["ux_designer"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=3,
        )

    @agent
    def critic(self) -> Agent:
        return Agent(
            config=self.agents_config["critic"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=2,  # critic은 짧고 날카롭게
        )

    @agent
    def developer(self) -> Agent:
        return Agent(
            config=self.agents_config["developer"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=4,
        )

    @agent
    def worker_developer(self) -> Agent:
        return Agent(
            config=self.agents_config["worker_developer"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=4,
        )

    @agent
    def code_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["code_reviewer"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=3,
        )

    @agent
    def qa_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_engineer"],
            llm=self._llm(),
            verbose=_verbose(),
            allow_delegation=False,
            max_iter=3,
        )

    # ──────────────────────────────────────────────
    # Tasks
    # ──────────────────────────────────────────────

    @task
    def plan(self) -> Task:
        return Task(config=self.tasks_config["plan"])

    @task
    def challenge_plan(self) -> Task:
        return Task(
            config=self.tasks_config["challenge_plan"],
            context=[self.plan()],
        )

    @task
    def design(self) -> Task:
        return Task(
            config=self.tasks_config["design"],
            context=[self.plan(), self.challenge_plan()],
        )

    @task
    def challenge_design(self) -> Task:
        return Task(
            config=self.tasks_config["challenge_design"],
            context=[self.design()],
        )

    @task
    def develop(self) -> Task:
        return Task(
            config=self.tasks_config["develop"],
            context=[self.plan(), self.design(), self.challenge_design()],
        )

    @task
    def develop_worker(self) -> Task:
        return Task(
            config=self.tasks_config["develop_worker"],
            context=[self.plan(), self.challenge_plan()],
        )

    @task
    def review(self) -> Task:
        return Task(
            config=self.tasks_config["review"],
            context=[self.develop(), self.develop_worker()],
        )

    @task
    def challenge_review(self) -> Task:
        return Task(
            config=self.tasks_config["challenge_review"],
            context=[self.review()],
        )

    @task
    def qa(self) -> Task:
        return Task(
            config=self.tasks_config["qa"],
            context=[
                self.develop(),
                self.develop_worker(),
                self.review(),
                self.challenge_review(),
            ],
        )

    @task
    def final_review(self) -> Task:
        return Task(
            config=self.tasks_config["final_review"],
            context=[
                self.plan(),
                self.develop(),
                self.develop_worker(),
                self.review(),
                self.challenge_review(),
                self.qa(),
            ],
        )

    # ──────────────────────────────────────────────
    # Crew
    # ──────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,   # @agent 데코레이터가 자동 수집
            tasks=self.tasks,     # @task 데코레이터가 자동 수집 (선언 순서 = 실행 순서)
            process=Process.sequential,
            verbose=_verbose(),
            memory=False,         # 태스크 간 컨텍스트는 context= 파라미터로 명시적 관리
                    output_log_file="crew_output.log",
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _verbose() -> bool:
    return os.environ.get("CREW_VERBOSE", "true").lower() != "false"

