(define (domain touch)
  (:requirements :strips :typing)
  (:types gripper shape)
  (:predicates
    (gripper_free ?g - gripper)
    (touched ?s - shape)
  )
  (:action make_contact
    :parameters (?g - gripper ?s - shape)
    :precondition (gripper_free ?g)
    :effect (touched ?s)
  )
)
