# check_lark_compiles.py
from lark import Lark

def check_lark_compiles():
    print("Checking if Lark compiles with start=domain")
    DOMAIN_GRAM = open("pddl_dom.lark").read()
    dom = Lark(DOMAIN_GRAM, start="domain", parser="lalr")
    print("✓ Compiles with start=domain")

    print("Checking if Lark compiles with start=problem")
    PROBLEM_GRAM = open("pddl_prob.lark").read()
    prb = Lark(PROBLEM_GRAM, start="problem", parser="lalr")
    print("✓ Compiles with start=problem")

def check_lark_on_example():
    
    gD = open("pddl_dom.lark").read()
    gP = open("pddl_prob.lark").read()
    dom_parser = Lark(gD, start="domain", parser="lalr")
    prb_parser = Lark(gP, start="problem", parser="lalr")

    DOMAIN = """
(define (domain blocksworld-hand)
(:requirements :strips :typing)
(:types block hand table)
(:predicates
    (on ?x - block ?y - block)
    (on-table ?x - block)
    (clear ?x - block)
    (holding ?x - block)
    (handempty ?h - hand)
)
(:action pickup
    :parameters (?x - block ?h - hand ?t - table)
    :precondition (and (on-table ?x) (clear ?x) (handempty ?h))
    :effect (and (not (on-table ?x)) (not (clear ?x)) (not (handempty ?h)) (holding ?x))
)
(:action putdown
    :parameters (?x - block ?h - hand ?t - table)
    :precondition (and (holding ?x))
    :effect (and (on-table ?x) (clear ?x) (handempty ?h) (not (holding ?x)))
)
(:action stack
    :parameters (?x - block ?y - block ?h - hand)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty ?h) (not (holding ?x)) (not (clear ?y)))
)
(:action unstack
    :parameters (?x - block ?y - block ?h - hand)
    :precondition (and (on ?x ?y) (clear ?x) (handempty ?h))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty ?h)))
)
)
    """

    PROBLEM = """
(define (problem bw-stack-d-on-c-on-b-on-a)
(:domain blocksworld-hand)
(:objects A B C D - block h - hand t - table)
(:init
    (on-table A) (on-table B) (on-table C) (on-table D)
    (clear A) (clear B) (clear C) (clear D)
    (handempty h)
)
(:goal (and (on D C) (on C B) (on B A)))
    )
    """

    dom_tree = dom_parser.parse(DOMAIN)
    prb_tree = prb_parser.parse(PROBLEM)
    print("✓ Domain parsed")
    print("✓ Problem parsed")


if __name__ == "__main__":
    check_lark_compiles()
    check_lark_on_example()