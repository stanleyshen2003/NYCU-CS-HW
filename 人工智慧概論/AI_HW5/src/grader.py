#!/usr/bin/env python3

import random, sys

from engine.const import Const
import graderUtil
import util
import collections
import copy

graderUtil.TOLERANCE = 1e-3
grader = graderUtil.Grader()
submission = grader.load('submission')

# General Notes:
# - Unless otherwise specified, all parts time out in 1 second.

############################################################
# Problem 1: Emission probabilities

def test1_1():
    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.030841805296, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00073380582967, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0269846478431, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0129150762582, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.00000261584106271, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.000924335357194, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0295673460685, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.000102360275238, ei.belief.getProb(5, 9))

grader.add_basic_part('part1-1', test1_1, 10, description="part1-1 test for emission probabilities")

def test1_2(): # test whether they put the pdf in the correct order
    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # be super rude to them! You can't swap a and c now!
      return a + b
    util.pdf = pdf

    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.012231949648, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00982248065925, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0120617259453, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0152083233155, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.0159738258744, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00989135100651, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0122435075636, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.018212043367, ei.belief.getProb(5, 9))
    util.pdf = oldpdf # replace the old pdf

grader.add_basic_part('part1-2', test1_2, 10, description="part1-2 test ordering of pdf")

############################################################
# Problem 2: Transition probabilities

def test2():
    ei = submission.ExactInference(30, 13)
    ei.elapseTime()
    grader.require_is_equal(0.0105778989624, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00250560512469, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0165024135157, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0178755550388, ei.belief.getProb(8, 4))

    ei.elapseTime()
    grader.require_is_equal(0.0138327373012, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00257237608713, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0232612833688, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0176501876956, ei.belief.getProb(8, 4))

grader.add_basic_part('part2', test2, 20, description="part2 test correctness of elapseTime()")

############################################################
# Problem 3: Particle filtering

def test3_1():
    random.seed(3)

    pf = submission.ParticleFilter(30, 13)

    pf.observe(555, 193, 800)
    grader.require_is_equal(0.02, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.04, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.94, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(8, 4))

    pf.observe(525, 193, 830)
    grader.require_is_equal(0.0, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(1.0, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(8, 4))


grader.add_basic_part('part3-1', test3_1, 10, description="part3-1 test for PF observe")

def test3_2():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.require_is_equal(69, len([k for k, v in list(pf.particles.items()) if v > 0])) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(58, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations

    grader.require_is_equal(6, pf.particles[(3, 9)])
    grader.require_is_equal(0, pf.particles[(2, 10)])
    grader.require_is_equal(3, pf.particles[(8, 4)])
    grader.require_is_equal(2, pf.particles[(12, 6)])
    grader.require_is_equal(2, pf.particles[(7, 8)])
    grader.require_is_equal(2, pf.particles[(11, 6)])
    grader.require_is_equal(0, pf.particles[(18, 7)])
    grader.require_is_equal(1, pf.particles[(20, 5)])

    pf.elapseTime()
    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(57, len([k for k, v in list(pf.particles.items()) if v > 0])) # Slightly more particles lie on the same (row, col) locations

    grader.require_is_equal(4, pf.particles[(3, 9)])
    grader.require_is_equal(0, pf.particles[(2, 10)]) # 0 --> 0
    grader.require_is_equal(5, pf.particles[(8, 4)])
    grader.require_is_equal(3, pf.particles[(12, 6)])
    grader.require_is_equal(0, pf.particles[(7, 8)])
    grader.require_is_equal(2, pf.particles[(11, 6)])
    grader.require_is_equal(0, pf.particles[(18, 7)]) # 0 --> 1
    grader.require_is_equal(1, pf.particles[(20, 5)]) # 1 --> 0

grader.add_basic_part('part3-2', test3_2, 10, description="part3-2 test for PF elapseTime")

def test3_3():
    random.seed(3)
    pf = submission.ParticleFilter(30, 13)
    grader.require_is_equal(69, len([k for k, v in list(pf.particles.items()) if v > 0])) # This should not fail unless your code changed the random initialization code.

    pf.elapseTime()
    grader.require_is_equal(58, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations
    pf.observe(555, 193, 800)

    grader.require_is_equal(200, sum(pf.particles.values())) # Do not lose particles
    grader.require_is_equal(2, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations
    grader.require_is_equal(0.025, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 6))
    grader.require_is_equal(0.975, pf.belief.getProb(22, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(22, 7))

    pf.elapseTime()
    grader.require_is_equal(4, len([k for k, v in list(pf.particles.items()) if v > 0])) # Most particles lie on the same (row, col) locations

    pf.observe(660, 193, 50)
    grader.require_is_equal(0.0, pf.belief.getProb(20, 4))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 5))
    grader.require_is_equal(0.0, pf.belief.getProb(21, 6))
    grader.require_is_equal(0.0, pf.belief.getProb(22, 6))
    grader.require_is_equal(1.0, pf.belief.getProb(22, 7))

grader.add_basic_part('part3-3', test3_3, 20, description="part3-3 test for PF observe AND elapseTime")

grader.grade()
