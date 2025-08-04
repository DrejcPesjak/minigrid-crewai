(define (domain touch_domain)
  (:requirements :strips :typing)
  (:types hand item)
  (:predicates (touched ?h - hand ?i - item))

  (:action touch_item
    :parameters (?h - hand ?i - item)
    :precondition ()
    :effect (touched ?h ?i)))
