(define (domain tabletop_touch_left)
 (:requirements :strips :typing)
 (:types item)
 (:predicates
  (touched_green_cube)
 )
 (:action touch_green_cube
  :parameters ()
  :precondition ()
  :effect (touched_green_cube)
 )
)