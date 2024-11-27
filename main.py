import FreeCAD
import Arch
import Fem
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from . import structureFication

report = [{"assumptions":[]}]

def check_ifc_not_related_to_rcc(doc):
    # List of keywords/values related to RCC (Reinforced Cement Concrete)
    rcc_related_keywords = [
        'concrete', 'steel', 'reinforced', 'rebar', 'steel reinforcement', 'cement',
        'RCC', 'reinforcement', 'FC30', 'Fe500', 'compression', 'tensile', 'strength',
        'concrete mix', 'load bearing', 'structural steel', 'slab', 'beam', 'column', 'footing'
    ]
    
    # Iterate through all the objects in the FreeCAD document
    for obj in doc.Objects:
        # Check if the object has an IFC property (this could be any property based on your model)
        if hasattr(obj, 'IfcProperties'):  # This assumes 'IfcProperties' exists for the object
            # Iterate through the IFC properties of the object
            for ifc_property in obj.IfcProperties:
                # Check if any IFC property value contains keywords related to RCC
                for keyword in rcc_related_keywords:
                    if ifc_property.lower().find(keyword) != -1:
                        return False  # If any related value is found, return False
    
    # If no related values were found in any objects, return True
    return True


def is_not_arch_object(doc):
    """
    Returns True if the object is NOT from the Arch Workbench.
    Checks if the object is a Wall, Slab, Beam, Column, or any other Arch object.
    """
    for obj in doc:
        # Check if the object is from the Arch Workbench by inspecting its type
        if isinstance(obj, Arch.BuildingPart) or isinstance(obj, Arch.Wall) or isinstance(obj, Arch.Slab) or isinstance(obj, Arch.Beam) or isinstance(obj, Arch.Column):
            return False  # Object is from the Arch Workbench
        
    return True # Object is not from the Arch Workbench

# RCC Advanced AI
class AdvancedRCCAI:
    def __init__(self):
        self.materials = {"concrete": "FC30", "steel": "Fe500"}
        self.codes = {"primary": "SANS", "secondary": "ACI"}
        self.results = {}
        self.ml_model = LinearRegression()
        self.training_data = {"X": [], "y": []}

    def identify_structures(self, doc):
        """
        Identifies RCC elements in the FreeCAD document.
        """
        structures = {"slabs": [], "beams": [], "columns": []}
        if is_not_arch_object(doc.objects):
            for obj in doc.objects:
                # Check if the object has the IFCType property
                if hasattr(obj, "IFCType"):
                    if obj.IFCType == "Slab":
                        structures["slabs"].append(obj)
                    elif obj.IFCType == "Beam":
                        structures["beams"].append(obj)
                    elif obj.IFCType == "Column":
                        structures["columns"].append(obj)
                    elif obj.IFCType == "Wall":
                        structures["walls"].append(obj)
                    elif obj.IFCType == "Footing":
                        structures["footings"].append(obj)
                    elif obj.IFCType == "Roof":
                        structures["roofs"].append(obj)
                    elif obj.IFCType == "Stair":
                        structures["stairs"].append(obj)
                    elif obj.IFCType == "Window":
                        structures["windows"].append(obj)
                    elif obj.IFCType == "Door":
                        structures["doors"].append(obj)
                    else:
                        structures["free_standing"].append(obj)  # For objects not recognized by IFCType
            self.results["structures"] = structures
            return structures
        else:
            for obj in doc.Objects:
                if Arch.isSlab(obj):
                    structures["slabs"].append(obj)
                elif Arch.isBeam(obj):
                    structures["beams"].append(obj)
                elif Arch.isColumn(obj):
                    structures["columns"].append(obj)
                elif Arch.isWall(obj):
                    structures["walls"].append(obj)
                elif Arch.isFooting(obj):
                    structures["footings"].append(obj)
                elif Arch.isRoof(obj):
                    structures["roofs"].append(obj)
                elif Arch.isStair(obj):
                    structures["stairs"].append(obj)
                elif Arch.isWindow(obj):
                    structures["windows"].append(obj)
                elif Arch.isDoor(obj):
                    structures["doors"].append(obj)
                else:
                    structures["free_standing"].append(obj)  # For objects not recognized by Arch Workbench
            self.results["structures"] = structures
            structureFication.update_training_data(structures)
            return structures

    def calculate_dead_load(self, obj):
        """
        Calculates dead load for a given RCC structure.
        """
        if hasattr(obj, "Shape") and obj.Shape.Volume:
            density = 25  # RCC density in kN/m³
            volume = obj.Shape.Volume / 1e9  # Convert mm³ to m³
            return density * volume
        return 0

    def setup_fem_analysis(self, obj):
        """
        Configures FEM analysis for a given structure.
        """
        if not Fem.isFemObject(obj):
            return None

        fem_analysis = Fem.Analysis.newAnalysis(obj)
        # Set up material properties, boundary conditions, etc.
        # For example: Add fixed supports, loads, and material definitions
        Fem.addFixedSupport(fem_analysis, obj)
        Fem.addLoad(fem_analysis, obj, load_value=10)  # Replace with actual load
        return fem_analysis

    def solve_fem(self, fem_analysis):
        """
        Solves the FEM analysis and retrieves results.
        """
        results = Fem.solve(fem_analysis)
        return {
            "bending_moment": results.get("bending_moment", 0),
            "shear_force": results.get("shear_force", 0),
            "deflection": results.get("deflection", 0),
        }

    def optimize_reinforcement(self, bending_moment):
        """
        Optimizes reinforcement design for a given bending moment.
        """
        def cost_function(reinforcement_area):
            # Define a cost function for reinforcement optimization
            cost_per_kg = 5  # Example cost per kg of steel
            steel_density = 7850  # kg/m³
            volume = reinforcement_area * 1  # Assuming 1m length for simplicity
            weight = volume * steel_density
            return cost_per_kg * weight

        constraints = [{"type": "ineq", "fun": lambda x: x - bending_moment / 250}]
        result = minimize(cost_function, x0=[0.01], constraints=constraints)
        return result.x[0] if result.success else 0

    def analyze_with_optimization(self, obj):
        """
        Performs advanced analysis with FEM and optimization.
        """
        dead_load = self.calculate_dead_load(obj)
        bending_moment = self.ml_model.predict([[dead_load]])[0]
        reinforcement_area = self.optimize_reinforcement(bending_moment)
        return {
            "dead_load": dead_load,
            "bending_moment": bending_moment,
            "optimized_reinforcement_area": reinforcement_area,
        }

    def generate_json_output(self, file_path):
        """
        Outputs results as a JSON file.
        """
        with open(file_path, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

    def run_analysis(self, doc):
        """
        Executes advanced RCC analysis.
        """
        self.train_ml_model()  # Train the ML model with FEM data
        structures = self.identify_structures(doc)

        # Analyze each structure
        analysis_results = {"slabs": [], "beams": [], "columns": []}
        for slab in structures["slabs"]:
            analysis_results["slabs"].append(self.analyze_with_optimization(slab))
        for beam in structures["beams"]:
            analysis_results["beams"].append(self.analyze_with_optimization(beam))
        for column in structures["columns"]:
            analysis_results["columns"].append(self.analyze_with_optimization(column))

        self.results["analysis"] = analysis_results
        return self.results

# Main Execution
def main():
    doc = FreeCAD.ActiveDocument
    if not doc:
        print("No active FreeCAD document found.")
        return
    
    if check_ifc_not_related_to_rcc(doc) and is_not_arch_object(doc):
        if input("Do you want AI to auto assign your structure type your (y/N)") is "y":
            print("AI will classifying your structure")
            structure_clasification_results = structureFication.classify_all_objects_with_ai()
            report.append({"structure_classification_results": structure_clasification_results})
        else:
            print("please edit the IFC Type of your structures")
    else:
        print("analyzing")

    analyzer = AdvancedRCCAI()
    results = analyzer.run_analysis(doc)
    report.append({"structure_analysis_results": results})
    output_path = "/home/sandile/Documents/RCC_advanced_analysis.json"
    analyzer.generate_json_output(output_path)

    print(f"Advanced analysis complete. Results saved to {output_path}.")

if __name__ == "__main__":
    main()
