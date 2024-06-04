import random
import numpy as np
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt

from typing import List, TypedDict
from matplotlib.animation import FuncAnimation, PillowWriter

class Member(TypedDict):
    chromosome: np.ndarray
    y_calc: List[float]
    fitness_err: float

class Problem(TypedDict):
    x_vec: np.ndarray
    y: float

class GeneticAlgorithm(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        # Parametros
        self.problem_df = pd.read_excel("data.xlsx", skiprows=1)
        
        self.title("Genetic Algorithm Y problem")
        self.geometry(f"{1366}x{768}")
        
        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(side="left", ipadx=10, ipady=10, fill="y", expand=False, padx=10, pady=10)
        
        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.pack(side="right", ipadx=10, ipady=10, fill="both", expand=True, padx=10, pady=10)
        
        self.problem_list: List[Problem] = []
        self.population: List[Member] = []
        
        self.initial_population = 20
        self.population_size = 50
        # self.generation = 100
        self.percent_of_elitism = 40
        self.mutation_rate = 70
        self.mutation_rate_gen = 70
        self.breakpoint_line = 0
        
        self.number_of_generations = 30
        
        # Collected data
        self.best_members: List[Member] = []
        self.best_members_error: List[float] = []
        self.worst_members: List[Member] = []
        self.median_values: List[float] = []
        self.y_values = []
        
        self.ax_pop_ev = None
        self.ax_des_apro = None
        self.ax_members = None
        
        # Initialize problem list
        self.extract_values(self.problem_df)
        
        self.run()
        
    def initization(self):
        # Initialize population
        for _ in range(self.initial_population):
            # Create member
            member = Member()
            
            # Assign random chromosome
            member["chromosome"] = [random.randint(0, 10) for _ in range(0, 5)]
            member["fitness_err"] = None
            member["y_calc"] = []
            
            # Append member to population
            self.population.append(member)
            
        # Calculate breakpoint line
        self.breakpoint_line = np.random.randint(1, len(self.population[0]["chromosome"]))

    def evalutatioon(self):
        # Evaluate population
        for member in self.population: # iter over population
            if member["fitness_err"] == None: # if member has not been evaluated
                member_err: List[float] = [] # create list to store error values
                for value in self.problem_list:
                    y = value["y"] # get y value
                    y_calc = member["chromosome"][0] + np.array([member["chromosome"][x + 1] * value["x_vec"][x] for x in range(0, 4)]).sum() # calculate y
                    member_err.append(y_calc - y)
                    member["y_calc"].append(y_calc)
                    
                member_err_array = np.array(member_err) # convert list to array
                member["fitness_err"] = np.linalg.norm(member_err_array) # calculate the error norm and assign to member
                
    def crossover(self):
        # Get the best member
        percent = int(self.population_size * self.percent_of_elitism / 100)
        best_member = self.population[:percent]
        
        index_mean = int(len(self.population) / 2)
        
        # kept the best members and the maximum of the population
        rest_of_population = self.population[:index_mean]
        
        # Calculate breakpoint line
        self.breakpoint_line = np.random.randint(1, len(self.population[0]["chromosome"]))
        
        # pairment and crossover
        for best in best_member:
            for rest in rest_of_population:
                # Create the first and second new members
                first_children = Member()
                seconde_children = Member()
                
                first_children["fitness_err"] = None
                seconde_children["fitness_err"] = None
                
                first_children["y_calc"] = []
                seconde_children["y_calc"] = []
                
                # Create the first and second new chromosomes
                first_children["chromosome"] = np.concatenate((best["chromosome"][:self.breakpoint_line], rest["chromosome"][self.breakpoint_line:]))
                seconde_children["chromosome"] = np.concatenate((rest["chromosome"][:self.breakpoint_line], best["chromosome"][self.breakpoint_line:]))
                
                # Append the new members to the population
                self.population.append(first_children)
                self.population.append(seconde_children)
                
        # remove the repeated members
        seen = set()
        self.population = [x for x in self.population if not (tuple(x["chromosome"]) in seen or seen.add(tuple(x["chromosome"])))]

    def mutation(self):
        # Mutate the childrens of the population
        for member in self.population:
            if member["fitness_err"] == None:
                if random.uniform(0, 100) < self.mutation_rate:
                    for i in range(len(member["chromosome"])):
                        if random.uniform(0, 100) < self.mutation_rate_gen:
                            member["chromosome"][i] += np.random.uniform(-2, 2) 
    
    def sort(self):
        # Sort population by fitness error
        self.population = sorted(self.population, key=lambda x: x["fitness_err"])
        
    def pode(self):
        self.sort()
        self.population = self.population[:self.population_size]
        
    def collect_data(self):
        self.best_members.append(self.population[0])
        self.median_values.append(np.median([m["fitness_err"] for m in self.population]))
        self.worst_members.append(self.population[-1])
        
    def make_video(self):
        fig_pop_ev = plt.figure(figsize=(10, 10))
        self.ax_pop_ev = fig_pop_ev.add_subplot(111)
        
        fig_des_aprox = plt.figure(figsize=(10, 10))
        self.ax_des_apro = fig_des_aprox.add_subplot(111)
        
        fig_members = plt.figure(figsize=(10, 10))
        self.ax_members = fig_members.add_subplot(111)
        
        print("making video")

        pop_ev_animation = FuncAnimation(fig_pop_ev, self.population_evo_update, frames=len(self.best_members), repeat=False)
        des_aprox_animation = FuncAnimation(fig_des_aprox, self.y_des_and_aproximate_update, frames=len(self.best_members), repeat=False)
        members_animation = FuncAnimation(fig_members, self.members_update, frames=len(self.best_members), repeat=False)
        
        print("complete video")
        
        pop_ev_animation.save("pop_ev.gif",  writer=PillowWriter(fps=2))
        des_aprox_animation.save("des_aprox.gif", writer=PillowWriter(fps=2))
        members_animation.save("members.gif",  writer=PillowWriter(fps=2))
        
    def population_evo_update(self, frame):
        self.ax_pop_ev.clear()
        
        self.ax_pop_ev.title.set_text("Population Evolution")
        self.ax_pop_ev.set_xlabel("Generation")
        self.ax_pop_ev.set_ylabel("Fitness Error")
        
        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))], [m["fitness_err"] for m in self.best_members[:frame]], label="Best members")
        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))], self.median_values[:frame], label="Median values")
        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))], [m["fitness_err"] for m in self.worst_members[:frame]], label="Worst members")
        self.ax_pop_ev.legend(labels=["Best members", "mean", "Worst members"])
    
    def y_des_and_aproximate_update(self, frame):
        self.ax_des_apro.clear()
        
        self.ax_des_apro.title.set_text("Y Desired and Aproximate")
        self.ax_des_apro.set_xlabel("member")
        self.ax_des_apro.set_ylabel("Y")
        
        self.ax_des_apro.plot([i for i in range(len(self.y_values))], self.y_values, label="Y Desired")
        self.ax_des_apro.plot([i for i in range(len(self.y_values))], self.best_members[frame]["y_calc"], label="Y Aproximate")

    def members_update(self, frame):
        self.ax_members.clear()
        
        self.ax_members.title.set_text("Members")
        self.ax_members.set_xlabel("member")
        self.ax_members.set_ylabel("Chromosome values")
        
        #A
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))], [m["chromosome"][0] for m in self.best_members[:frame]], label="A")
        #B
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))], [m["chromosome"][1] for m in self.best_members[:frame]], label="B")
        #C
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))], [m["chromosome"][2] for m in self.best_members[:frame]], label="C")
        #D
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))], [m["chromosome"][3] for m in self.best_members[:frame]], label="D")
        #E
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))], [m["chromosome"][4] for m in self.best_members[:frame]], label="E")
        
        self.ax_members.legend(labels=["A", "B", "C", "D", "E"])

    def run(self):
        self.initization()
        self.evalutatioon()
        self.sort()
        
        gen = 0
        while gen < self.number_of_generations:
            self.crossover()
            self.mutation()
            self.evalutatioon()
            self.sort()
            self.collect_data()
            self.pode()
            
            print("generation " + str(gen))
            gen += 1

            print("\n best member error norm: ")
            print(self.population[0]["fitness_err"])
        self.make_video()
    
    def extract_values(self, df: pd.DataFrame):
        x_values = self.problem_df[['x1', 'x2', 'x3', 'x4']].values.tolist()
        self.y_values = self.problem_df["y"]
        
        for y in self.y_values:
            problem = Problem()
            problem["x_vec"] = x_values.pop(0)
            problem["y"] = y
            
            self.problem_list.append(problem)

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.mainloop()
    # ga.run()