(define (domain doorkey_highlevel)
  (:requirements :strips :typing)
  (:types agent door region)
  (:predicates (door_opened) (at_goal ?a - agent))
  (:action ensure_door_open
    :parameters (?a - agent ?d - door)
    :precondition ()
    :effect (door_opened))
  (:action finish
    :parameters (?a - agent ?g - region)
    :precondition (door_opened)
    :effect (at_goal ?a)))