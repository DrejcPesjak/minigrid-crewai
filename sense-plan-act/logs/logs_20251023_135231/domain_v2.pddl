(define (domain unlock_pickup_domain)
 (:requirements :strips :typing)
 (:types agent key door box region)
 (:predicates
  (have_key)
  (door_open)
  (have_box)
 )
 (:action pick_up_obj
  :parameters (?a - agent ?k - key)
  :precondition ()
  :effect (have_key)
 )
 (:action open_yellow_door
  :parameters (?a - agent)
  :precondition (have_key)
  :effect (door_open)
 )
 (:action pick_up
  :parameters (?a - agent ?b - box ?r - region)
  :precondition (door_open)
  :effect (have_box)
 )
)