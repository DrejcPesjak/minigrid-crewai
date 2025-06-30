(define (domain pickup_only_domain)
  (:requirements :strips :typing)
  (:types agent item)
  (:predicates
    (available ?i - item)
    (holding ?a - agent ?i - item)
  )
  (:action pickup_only_v3
    :parameters (?a - agent ?i - item)
    :precondition (available ?i)
    :effect (holding ?a ?i)
  )
)