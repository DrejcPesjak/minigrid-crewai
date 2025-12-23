(define (domain mini_simple_crossing_s11n5)
  (:requirements :strips :typing)
  (:types agent location)
  (:constants start_loc crossing_loc goal_loc - location)
  (:predicates (at ?a - agent ?l - location))
  (:action goto_crossing
    :parameters (?a - agent)
    :precondition (at ?a start_loc)
    :effect (at ?a crossing_loc))
  (:action goto_goal
    :parameters (?a - agent)
    :precondition (at ?a crossing_loc)
    :effect (at ?a goal_loc)))