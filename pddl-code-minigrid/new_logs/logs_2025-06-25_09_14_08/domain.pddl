(define (domain goto_local)
 (:requirements :strips :typing)
  (:types agent)
   (:predicates (is_agent ?a - agent) (at_goal))
    (:action reach_goal_v3
      :parameters (?a - agent)
        :precondition (is_agent ?a)
          :effect (at_goal)))