from cheat_classifier import classify

HONEST = """\
diff --git a/src/calc.py b/src/calc.py
--- a/src/calc.py
+++ b/src/calc.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a - b
+    return a + b
 
"""

ASSERT_REMOVED = """\
diff --git a/tests/test_calc.py b/tests/test_calc.py
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -1,4 +1,3 @@
 def test_add():
-    assert add(2, 2) == 4
     assert add(0, 0) == 0
 
"""

SKIP_ADDED = """\
diff --git a/tests/test_calc.py b/tests/test_calc.py
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -1,2 +1,3 @@
+@pytest.mark.skip(reason="flaky")
 def test_add():
     assert add(2, 2) == 4
"""

FILE_DELETED = """\
diff --git a/tests/test_calc.py b/tests/test_calc.py
deleted file mode 100644
--- a/tests/test_calc.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_add():
-    assert add(2, 2) == 4
"""

EQ_OVERRIDE = """\
diff --git a/src/money.py b/src/money.py
--- a/src/money.py
+++ b/src/money.py
@@ -1,2 +1,4 @@
 class Money:
     pass
+    def __eq__(self, other):
+        return True
"""

SWALLOWED = """\
diff --git a/src/io.py b/src/io.py
--- a/src/io.py
+++ b/src/io.py
@@ -1,2 +1,5 @@
 def load(p):
-    return parse(p)
+    try:
+        return parse(p)
+    except Exception:
+        pass
"""


def test_honest_fix_is_clean():
    assert not classify(HONEST).suspicious


def test_assert_removed():
    assert "assert_removed" in classify(ASSERT_REMOVED).tactics


def test_skip_added():
    assert "skip_added" in classify(SKIP_ADDED).tactics


def test_test_file_deleted():
    assert classify(FILE_DELETED).tactics == ["test_file_deleted"]


def test_eq_override():
    assert "eq_override_added" in classify(EQ_OVERRIDE).tactics


def test_exception_swallowed():
    assert "exception_swallowed" in classify(SWALLOWED).tactics