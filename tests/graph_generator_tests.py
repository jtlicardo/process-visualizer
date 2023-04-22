import unittest
import os, sys
import json

sys.path.insert(1, "src")

from graph_generator import GraphGenerator


class GraphGeneratorTests(unittest.TestCase):
    def test_generate_graph(self):
        for file in os.listdir("tests/bpmn_structures"):
            file_path = os.path.join("tests/bpmn_structures", file)
            with open("tests/bpmn_structures/" + file, "r") as f:
                file = f.read()
            data = json.loads(file)
            graph_generator = GraphGenerator(data=data, test_mode=True)
            try:
                graph_generator.generate_graph()
            except Exception as e:
                print(f"Error occurred in file: {file_path}")
                print(f"Error message: {e}")
                raise


if __name__ == "__main__":
    unittest.main()
