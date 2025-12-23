(define (domain babyai_pickup)
  (:requirements :strips :typing)
  (:types agent item)
  (:predicates
    (adjacent ?i - item)
    (holding ?i - item)
  )
  (:action nav_to
    :parameters (?a - agent ?i - item)
    :precondition ()
    :effect (adjacent ?i)
  )
  (:action pick_up_item
    :parameters (?a - agent ?i - item)
    :precondition (adjacent ?i)
    :effect (holding ?i)
  )
)