(define (domain tabletop_touch_far_cube)
 (:requirements :strips :typing)
 (:types item)
 (:predicates
  (at_red_cube)
  (touched_red_cube)
 )
 (:action move_to_red_cube
  :parameters ()
  :precondition ()
  :effect (at_red_cube)
 )
 (:action touch_red_cube
  :parameters ()
  :precondition (at_red_cube)
  :effect (touched_red_cube)
 )
)