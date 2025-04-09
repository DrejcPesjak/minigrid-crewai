(define (domain cupcake-domain)
  (:predicates
    (gripper-empty)
    (cupcake-on-table)
    (cupcake-on-plate)
    (holding-cupcake)
  )

  (:action pickup-cupcake
    :parameters ()
    :precondition (and (gripper-empty) (cupcake-on-table))
    :effect (and (holding-cupcake) (not (cupcake-on-table)) (not (gripper-empty)))
  )

  (:action place-cupcake-on-plate
    :parameters ()
    :precondition (holding-cupcake)
    :effect (and (cupcake-on-plate) (gripper-empty) (not (holding-cupcake)))
  )
)