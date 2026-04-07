from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, student_id, name, age, grade_level):
        self.student_id = student_id
        self.name = name
        self.age = age
        self.grade_level = grade_level
        self.grades = {}  # Stores subject: grade

    def add_grade(self, subject, grade):
        self.grades[subject] = grade
        print(f"Added grade {grade} for {subject} to {self.name}.")

    def get_average_grade(self):
        if not self.grades:
            return 0
        return sum(self.grades.values()) / len(self.grades)

    def display_info(self):
        print(f"ID: {self.student_id}, Name: {self.name}, Age: {self.age}, Grade Level: {self.grade_level}")
        print("Grades:", self.grades)
        print(f"Average Grade: {self.get_average_grade():.2f}\n")


class School:
    def __init__(self, name):
        self.name = name
        self.students = {}

    def add_student(self, student):
        if student.student_id in self.students:
            print(f"Student with ID {student.student_id} already exists.")
        else:
            self.students[student.student_id] = student
            print(f"Added student {student.name} to {self.name}.")

    def remove_student(self, student_id):
        if student_id in self.students:
            removed_student = self.students.pop(student_id)
            print(f"Removed student {removed_student.name} from {self.name}.")
        else:
            print(f"No student found with ID {student_id}.")

    def display_all_students(self):
        print(f"All students in {self.name}:")
        for student in self.students.values():
            student.display_info()


# Example Usage
if __name__ == "__main__":
    # Create School
    my_school = School("Greenwood High")

    # Create Students
    student1 = Student(1, "Alice", 14, 9)
    student2 = Student(2, "Bob", 15, 10)

    # Add grades
    student1.add_grade("Math", 95)
    student1.add_grade("English", 88)
    student2.add_grade("Math", 78)
    student2.add_grade("Science", 85)

    # Add students to school
    my_school.add_student(student1)
    my_school.add_student(student2)

    # Display all students
    my_school.display_all_students()

    # Remove a student
    my_school.remove_student(1)

    # Display remaining students
    my_school.display_all_students()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])