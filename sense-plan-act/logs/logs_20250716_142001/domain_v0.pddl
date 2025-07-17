(define (domain gotolocal)
 (:requirements :strips :typing)
 (:types region)
 (:predicates (at ?r - region))
 (:action move
  :parameters (?from - region ?to - region)
  :precondition (at ?from)
  :effect (and (not (at ?from)) (at ?to))))