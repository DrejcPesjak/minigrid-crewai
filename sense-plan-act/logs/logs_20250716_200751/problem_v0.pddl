(define (problem red_blue_doors_problem)
 (:domain red_blue_doors)
 (:objects
  agent1 - agent
  red_door blue_door - door)
 (:init
  (is_red red_door)
  (is_blue blue_door)
 )
 (:goal (and (door_open red_door) (door_open blue_door)))
)