(define (domain mini_empty_navigation)
 (:requirements :strips :typing)
 (:types agent goal)
 (:predicates (at_goal) (alive ?a - agent))
 (:action reach_goal
  :parameters (?a - agent ?g - goal)
  :precondition (alive ?a)
  :effect (at_goal)))