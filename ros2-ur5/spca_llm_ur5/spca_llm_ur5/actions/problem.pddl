(define (problem touch_blue_cube_problem)
  (:domain touch_domain)
  (:objects
    gripper - hand
    cube_blue cube_green cube_red circle_blue circle_red circle_green - item)
  (:init)
  (:goal (touched gripper cube_blue)))
