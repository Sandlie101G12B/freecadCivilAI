import FreeCAD as App
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_training_data():
    with open('raw_structure_training_data.json', 'r') as file:
        raw_training_data = json.load(file)
        data = []
        for category, dimentions in raw_training_data:
            data["dimentions"].append(dimentions)
            data["category"].append(category)
        return data

def store_training_data(training_data):

    print('s')

def update_training_data(structures):
    training_data = []
    
    # Iterate through the classified structures
    for category, objs in structures.items():
        for obj in objs:
            # Check if the object has the Shape property (this is where the geometry data is stored)
            if hasattr(obj, 'Shape'):
                if hasattr(obj, 'IFCType') and obj.IFCType == category[:-1]: # If both the arch workbench and IFCType of the object are the same.
                    # Get the bounding box dimensions (X, Y, Z) of the object
                    bbox = obj.Shape.BoundBox
                    dimensions = [bbox.XLength, bbox.YLength, bbox.ZLength]
                    
                    # Append the structure type and its dimensions to the training data list
                    training_data.append([category[:-1], dimensions])
    
    store_training_data(training_data)

# Sample training data (dimensions: length, width, height) and labels
structure_training_data = load_training_data()
structure_clasification_data = np.array(structure_training_data["dimentions"])
labels = structure_training_data["category"]

# Classification colors
CLASS_COLORS = {
    "Slab": (0.5, 0.5, 1.0),    # Light blue
    "Column": (1.0, 0.5, 0.5),  # Light red
    "Beam": (0.5, 1.0, 0.5),    # Light green
    "Wall": (0.9, 0.9, 0.5),    # Light yellow
    "Door": (1.0, 1.0, 0.5),    # Light yellow (for doors)
    "Window": (0.5, 1.0, 1.0),  # Light cyan (for windows)
    "Roof": (0.8, 0.5, 0.2),    # Light brown
    "Stair": (0.8, 0.6, 0.6),   # Light pink
    "Ramp": (0.6, 0.8, 0.6),    # Light greenish
    "Footing": (0.3, 0.3, 0.3), # Dark gray (for foundation/footing)
    "Foundation": (0.2, 0.2, 0.2), # Darker gray for foundation elements
    "BuildingElementProxy": (0.8, 0.8, 0.8),  # Gray (for non-categorized elements)
    "CurtainWall": (0.4, 0.7, 0.9),  # Light blue (for curtain walls)
    "Pillar": (0.9, 0.7, 0.3),  # Golden yellow (for pillars)
    "Truss": (0.6, 0.3, 0.1),   # Dark brown
    "SlabElement": (0.5, 0.5, 1.0), # Same as slab (for slab elements)
    "Opening": (0.6, 0.6, 0.6), # Light gray for openings
    "Pipe": (0.9, 0.4, 0.4),   # Light red for pipes
    "CableTray": (0.4, 0.7, 0.4), # Light green for cable trays
    "LightFixture": (1.0, 1.0, 1.0), # White for light fixtures
}

# IFC type mapping
IFC_TYPES = {
    "Slab": "IfcSlab",
    "Column": "IfcColumn",
    "Beam": "IfcBeam",
    "Wall": "IfcWall",
    "Door": "IfcDoor",
    "Window": "IfcWindow",
    "Roof": "IfcRoof",
    "Stair": "IfcStair",
    "Ramp": "IfcRamp",
    "Footing": "IfcFooting",
    "Foundation": "IfcFoundation",
    "BuildingElementProxy": "IfcBuildingElementProxy",  # For elements that don't fall into other categories
    "CurtainWall": "IfcCurtainWall",
    "Pillar": "IfcPillar",  # Custom IFC type, if applicable
    "Truss": "IfcTruss",  # If you are modeling trusses
    "SlabElement": "IfcSlabElement",  # For slab elements that aren't purely slab
    "Opening": "IfcOpeningElement",  # For openings in walls, slabs, etc.
    "Pipe": "IfcPipeSegment",  # For pipe elements, if included
    "CableTray": "IfcCableSegment",  # For cable tray elements, if included
    "LightFixture": "IfcLightFixture",  # If your project includes lighting elements
}

# Train a Random Forest Classifier
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(structure_clasification_data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained with accuracy: {model.score(X_test, y_test) * 100:.2f}%")
    return model

# Classification function using AI model
def classify_with_ai(obj, model):
    """
    Classifies a FreeCAD object using an AI model.

    Args:
        obj: FreeCAD object to classify.
        model: Trained AI model.

    Returns:
        str: Predicted classification (e.g., 'Slab', 'Column', 'Beam', 'Other').
    """
    bbox = obj.Shape.BoundBox
    dimensions = [bbox.XLength, bbox.YLength, bbox.ZLength]
    prediction = model.predict([dimensions])
    return prediction[0]

# Main function to classify all Part Design objects
def classify_all_objects_with_ai():
    """
    Classifies all Part Design objects in the active FreeCAD document using AI.
    """
    doc = App.ActiveDocument
    
    model = train_model()  # Train the AI model
    results = []
    print("Classifying objects using AI...")

    for obj in doc.Objects:
        if hasattr(obj, "Shape") and obj.Shape.Solids:
            classification = classify_with_ai(obj, model)

            # Update the IFC type
            if hasattr(obj, "IfcType"):
                obj.IfcType = IFC_TYPES[classification]
            else:
                setattr(obj, "IfcType", IFC_TYPES[classification])

            # Change color based on classification
            if hasattr(obj.ViewObject, "ShapeColor"):
                obj.ViewObject.ShapeColor = CLASS_COLORS[classification]
            classification += classification + "s"
            # Add to results
            results.append({
                "Name": obj.Name,
                "Dimensions": {
                    "Length": obj.Shape.BoundBox.XLength,
                    "Width": obj.Shape.BoundBox.YLength,
                    "Height": obj.Shape.BoundBox.ZLength
                },
                "Classification": classification,
                "IfcType": obj.IfcType
            })
        else:
            results.append({
                "Name": obj.Name,
                "Error": "Not a solid or invalid shape"
            })

    # Print and return results in JSON format
    results_json = json.dumps(results, indent=4)
    print(results_json)
    return results_json

# Run the AI-powered classification
classify_all_objects_with_ai()
