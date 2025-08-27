(define (problem touch_middle_cube_problem)
  (:domain touch)
  (:objects
    gripper1 - gripper
    middle_cube - shape
  )
  (:init
    (gripper_free gripper1)
  )
  (:goal (touched middle_cube))
)
