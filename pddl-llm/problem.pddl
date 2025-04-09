(define (problem cupcake-problem)
  (:domain cupcake-domain)
  (:objects
    gripper - robot
    the-cupcake - cupcake-type
    the-plate - plate-type
  )

  (:init
    (empty gripper)
    (on-table the-cupcake)
  )

  (:goal (on-plate the-cupcake))
)