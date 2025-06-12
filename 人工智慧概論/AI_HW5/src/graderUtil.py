"""
Library to do grading of Python programs.
Usage (see grader.py):

    # create a grader
    grader = Grader("Name of assignment")

    # add a basic test
    grader.addBasicPart(number, grade_func, max_points, max_seconds, description="a basic test")

    # add a hidden test
    grader.addHiddenPart(number, grade_func, max_points, max_seconds, description="a hidden test")

    # add a manual grading part
    grader.addManualPart(number, grade_func, max_points, description="written problem")

    # run grading
    grader.grade()
"""

import argparse
import datetime
import gc
import json
import os
import signal
import sys
import traceback

default_max_seconds = 5  # 5 second
TOLERANCE = 1e-4  # For measuring whether two floats are equal

BASIC_MODE = 'basic'  # basic
AUTO_MODE = 'auto'  # basic + hidden
ALL_MODE = 'all'  # basic + hidden + manual


# When reporting stack traces as feedback, ignore parts specific to the grading
# system.
def is_traceback_item_grader(item):
    return item[0].endswith('graderUtil.py')


def is_collection(x):
    return isinstance(x, list) or isinstance(x, tuple)


# Return whether two answers are equal.
def is_equal(true_answer, pred_answer, tolerance=TOLERANCE):
    # Handle floats specially
    if isinstance(true_answer, float) or isinstance(pred_answer, float):
        return abs(true_answer - pred_answer) < tolerance
    # Recurse on collections to deal with floats inside them
    if is_collection(true_answer) and is_collection(pred_answer) and len(true_answer) == len(pred_answer):
        for a, b in zip(true_answer, pred_answer):
            if not is_equal(a, b):
                return False
        return True
    if isinstance(true_answer, dict) and isinstance(pred_answer, dict):
        if len(true_answer) != len(pred_answer):
            return False
        for k, v in list(true_answer.items()):
            if not is_equal(pred_answer.get(k), v):
                return False
        return True

    # Numpy array comparison
    if type(true_answer).__name__ == 'ndarray':
        import numpy as np
        if isinstance(true_answer, np.ndarray) and isinstance(pred_answer, np.ndarray):
            if true_answer.shape != pred_answer.shape:
                return False
            for a, b in zip(true_answer, pred_answer):
                if not is_equal(a, b):
                    return False
            return True

    # Do normal comparison
    return true_answer == pred_answer


# Run a function, timing out after max_seconds.
class TimeoutFunctionException(Exception):
    pass


class TimeoutFunction:
    def __init__(self, function, max_seconds):
        self.max_seconds = max_seconds
        self.function = function

    @staticmethod
    def handle_max_seconds(signum, frame):
        print('TIMEOUT!')
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if os.name == 'nt':
            # Windows does not have signal.SIGALRM
            # Will not stop after max_seconds second but can still throw an exception
            time_start = datetime.datetime.now()
            result = self.function(*args)
            time_end = datetime.datetime.now()
            if time_end - time_start > datetime.timedelta(seconds=self.max_seconds + 1):
                raise TimeoutFunctionException()
            return result
            # End modification for Windows here
        signal.signal(signal.SIGALRM, self.handle_max_seconds)
        signal.alarm(self.max_seconds + 1)
        result = self.function(*args)
        signal.alarm(0)
        return result


class Part:
    def __init__(self, number, grade_func, max_points, max_seconds, extra_credit, description, basic):
        if not isinstance(number, str):
            raise Exception("Invalid number: %s" % number)
        if grade_func is not None and not callable(grade_func):
            raise Exception("Invalid grade_func: %s" % grade_func)
        if not isinstance(max_points, int):
            raise Exception("Invalid max_points: %s" % max_points)
        if max_seconds is not None and not isinstance(max_seconds, int):
            raise Exception("Invalid max_seconds: %s" % max_seconds)
        if not description:
            print('ERROR: description required for part {}'.format(number))
        # Specification of part
        self.number = number  # Unique identifier for this part.
        self.description = description  # Description of this part
        self.grade_func = grade_func  # Function to call to do grading
        self.max_points = max_points  # Maximum number of points attainable on this part
        self.max_seconds = max_seconds  # Maximum allowed time that the student's code can take (in seconds)
        self.extra_credit = extra_credit  # Whether this is an extra credit problem
        self.basic = basic
        # Grading the part
        self.points = 0
        self.side = None  # Side information
        self.seconds = 0
        self.messages = []
        self.failed = False

    def fail(self):
        self.failed = True

    def is_basic(self):
        return self.grade_func is not None and self.basic

    def is_hidden(self):
        return self.grade_func is not None and not self.basic

    def is_auto(self):
        return self.grade_func is not None

    def is_manual(self):
        return self.grade_func is None


class Grader:
    def __init__(self, args=None):
        if args is None:
            args = sys.argv
        self.parts = []  # Parts (to be added)
        self.useSolution = False  # Set to true if we are actually evaluating the hidden test cases

        parser = argparse.ArgumentParser()
        parser.add_argument('--js', action='store_true', help='Write JS file with information about this assignment')
        parser.add_argument('--json', action='store_true',
                            help='Write JSON file with information about this assignment')
        parser.add_argument('--summary', action='store_true', help='Don\'t actually run code')
        parser.add_argument('remainder', nargs=argparse.REMAINDER)
        self.params = parser.parse_args(args[1:])

        args = self.params.remainder
        if len(args) < 1:
            self.mode = AUTO_MODE
            self.selectedPartName = None
        else:
            if args[0] in [BASIC_MODE, AUTO_MODE, ALL_MODE]:
                self.mode = args[0]
                self.selectedPartName = None
            else:
                self.mode = AUTO_MODE
                self.selectedPartName = args[0]

        self.messages = []  # General messages
        self.currentPart = None  # Which part we're grading
        self.fatalError = False  # Set this if we should just stop immediately

    def add_basic_part(self, number, grade_func, max_points=1, max_seconds=default_max_seconds, extra_credit=False,
                       description=""):
        """Add a basic test case. The test will be visible to students."""
        self.assert_new_number(number)
        part = Part(number, grade_func, max_points, max_seconds, extra_credit, description, basic=True)
        self.parts.append(part)

    def add_hidden_part(self, number, grade_func, max_points=1, max_seconds=default_max_seconds, extra_credit=False,
                        description=""):
        """Add a hidden test case. The output should NOT be visible to students
        and so should be inside a BEGIN_HIDE block."""
        self.assert_new_number(number)
        part = Part(number, grade_func, max_points, max_seconds, extra_credit, description, basic=False)
        self.parts.append(part)

    def add_manual_part(self, number, max_points, extra_credit=False, description=""):
        """Add a manual part."""
        self.assert_new_number(number)
        part = Part(number, None, max_points, None, extra_credit, description, basic=False)
        self.parts.append(part)

    def assert_new_number(self, number):
        if number in [part.number for part in self.parts]:
            raise Exception("Part number %s already exists" % number)

    # Try to load the module (submission from student).
    def load(self, module_name):
        try:
            return __import__(module_name)
        except Exception as e:
            self.fail("Threw exception when importing '%s': %s" % (module_name, e))
            self.fatalError = True
            return None
        except:
            self.fail("Threw exception when importing '%s'" % module_name)
            self.fatalError = True
            return None

    def grade_part(self, part):
        print('----- START PART %s%s: %s' % (
            part.number, ' (extra credit)' if part.extra_credit else '', part.description))
        self.currentPart = part

        start_time = datetime.datetime.now()
        try:
            TimeoutFunction(part.grade_func, part.max_seconds)()  # Call the part's function
        except KeyboardInterrupt:
            raise
        except MemoryError:
            signal.alarm(0)
            gc.collect()
            self.fail('Memory limit exceeded.')
        except TimeoutFunctionException:
            signal.alarm(0)
            self.fail('Time limit (%s seconds) exceeded.' % part.max_seconds)
        except Exception as e:
            signal.alarm(0)
            self.fail('Exception thrown: %s -- %s' % (str(type(e)), str(e)))
            self.print_exception()
        except SystemExit:
            # Catch SystemExit raised by exit(), quit() or sys.exit()
            # This class is not a subclass of Exception and we don't
            # expect students to raise it.
            self.fail('Unexpected exit.')
            self.print_exception()
        end_time = datetime.datetime.now()
        part.seconds = (end_time - start_time).seconds
        if part.is_hidden() and not self.useSolution:
            display_points = '???/%s points (hidden test ungraded)' % part.max_points
        else:
            display_points = '%s/%s points' % (part.points, part.max_points)
        print('----- END PART %s [took %s (max allowed %s seconds), %s]' % (
            part.number, end_time - start_time, part.max_seconds, display_points))
        print()

    def get_selected_parts(self):
        parts = []
        for part in self.parts:
            if self.selectedPartName is not None and self.selectedPartName != part.number:
                continue
            if self.mode == BASIC_MODE:
                if part.is_basic():
                    parts.append(part)
            elif self.mode == AUTO_MODE:
                if part.is_auto():
                    parts.append(part)
            elif self.mode == ALL_MODE:
                parts.append(part)
            else:
                raise Exception("Invalid mode: {}".format(self.mode))
        return parts

    def grade(self):
        parts = self.get_selected_parts()

        result = {'mode': self.mode}

        # Grade it!
        if not self.params.summary and not self.fatalError:
            print('========== START GRADING')
            for part in parts:
                self.grade_part(part)

            # When students have it (not useSolution), only include basic tests.
            active_parts = [part for part in parts if self.useSolution or part.basic]

            total_points = sum(part.points for part in active_parts if not part.extra_credit)
            extra_credit = sum(part.points for part in active_parts if part.extra_credit)
            max_total_points = sum(part.max_points for part in active_parts if not part.extra_credit)
            max_extra_credit = sum(part.max_points for part in active_parts if part.extra_credit)

            # if not self.useSolution:
                # print('Note that the hidden test cases do not check for correctness.'
                      # '\nThey are provided for you to verify that the functions '
                      # 'do not crash and run within the time limit.'
                      # '\nPoints for these parts not assigned by the grader (indicated by "--").')
            print('========== END GRADING [%d/%d points + %d/%d extra credit]' %
                  (total_points, max_total_points, extra_credit, max_extra_credit))

        result_parts = []
        leaderboard = []
        for part in parts:
            r = {'number': part.number, 'name': part.description}

            if self.params.summary:
                # Just print out specification of the part
                r['description'] = part.description
                r['max_seconds'] = part.max_seconds
                r['max_points'] = part.max_points
                r['extra_credit'] = part.extra_credit
                r['basic'] = part.basic
            else:
                r['score'] = part.points
                # Force max_score to be 0 for extra credits for displaying correct total points on Gradescope
                r['max_score'] = 0 if (part.extra_credit and self.mode == AUTO_MODE) else part.max_points
                r["visibility"] = "after_published" if part.is_hidden() else "visible"
                r['seconds'] = part.seconds
                if part.side is not None:
                    r['side'] = part.side
                r['output'] = "\n".join(part.messages)

                if part.side is not None:
                    for k in part.side:
                        leaderboard.append({"name": k, "value": part.side[k]})
            result_parts.append(r)
        result['tests'] = result_parts
        result['leaderboard'] = leaderboard

        self.output(self.mode, result)

        def display(name, select_extra_credit):
            parts_to_display = [p for p in self.parts if p.extra_credit == select_extra_credit]
            max_basic_points = sum(p.max_points for p in parts_to_display if p.is_basic())
            max_hidden_points = sum(p.max_points for p in parts_to_display if p.is_hidden())
            max_manual_points = sum(p.max_points for p in parts_to_display if p.is_manual())
            max_total_points_found = max_basic_points + max_hidden_points + max_manual_points
            print("Total %s (basic auto/coding + hidden auto/coding + manual/written): %d + %d + %d = %d" %
                  (name, max_basic_points, max_hidden_points, max_manual_points, max_total_points_found))
            if not select_extra_credit and max_total_points_found != 75:
                print('WARNING: max_total_points = {} is not 75'.format(max_total_points_found))

        if self.params.summary:
            display('points', False)
            display('extra credit', True)

    def output(self, mode, result):
        if self.params.json:
            path = 'grader-{}.json'.format(mode)
            with open(path, 'w') as out:
                print(json.dumps(result), file=out)
            print('Wrote to %s' % path)
        if self.params.js:
            path = 'grader-{}.js'.format(mode)
            with open(path, 'w') as out:
                print('var ' + mode + 'Result = ' + json.dumps(result) + ';', file=out)
            print('Wrote to %s' % path)

    # Called by the grader to modify state of the current part

    def add_points(self, amt):
        self.currentPart.points += amt

    def assign_full_credit(self):
        if not self.currentPart.failed:
            self.currentPart.points = self.currentPart.max_points
        return True

    def assign_partial_credit(self, credit):
        self.currentPart.points = credit
        return True

    def set_side(self, side):
        self.currentPart.side = side

    @staticmethod
    def truncate_string(string, length=200):
        if len(string) <= length:
            return string
        else:
            return string[:length] + '...'

    def require_is_numeric(self, answer):
        if isinstance(answer, int) or isinstance(answer, float):
            return self.assign_full_credit()
        else:
            return self.fail("Expected either int or float, but got '%s'" % self.truncate_string(answer))

    def require_is_one_of(self, true_answers, pred_answer):
        if pred_answer in true_answers:
            return self.assign_full_credit()
        else:
            return self.fail("Expected one of %s, but got '%s'" % (
                self.truncate_string(true_answers), self.truncate_string(pred_answer)))

    def require_is_equal(self, true_answer, pred_answer, tolerance=TOLERANCE):
        if is_equal(true_answer, pred_answer, tolerance):
            return self.assign_full_credit()
        else:
            return self.fail("Expected '%s', but got '%s'" % (
                self.truncate_string(str(true_answer)), self.truncate_string(str(pred_answer))))

    def require_is_less_than(self, less_than_quantity, pred_answer):
        if pred_answer < less_than_quantity:
            return self.assign_full_credit()
        else:
            return self.fail("Expected to be < %f, but got %f" % (less_than_quantity, pred_answer))

    def require_is_greater_than(self, greater_than_quantity, pred_answer):
        if pred_answer > greater_than_quantity:
            return self.assign_full_credit()
        else:
            return self.fail("Expected to be > %f, but got %f" %
                             (greater_than_quantity, pred_answer))

    def require_is_true(self, pred_answer):
        if pred_answer:
            return self.assign_full_credit()
        else:
            return self.fail("Expected to be true, but got false")

    def fail(self, message):
        print('FAIL:', message)
        self.add_message(message)
        if self.currentPart:
            self.currentPart.points = 0
            self.currentPart.fail()
        return False

    def print_exception(self):
        tb = [item for item in traceback.extract_tb(sys.exc_info()[2]) if not is_traceback_item_grader(item)]
        for item in traceback.format_list(tb):
            self.fail('%s' % item)

    def add_message(self, message):
        if not self.useSolution:
            print(message)
        if self.currentPart:
            self.currentPart.messages.append(message)
        else:
            self.messages.append(message)
