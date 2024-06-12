import random
import threading
import time

import numpy as np
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
from tkinter import PhotoImage
from PIL import Image, ImageTk

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

        self.title("Genetic Algorithm Y problem")
        self.geometry(f"{1366}x{768}")

        self.options_frame = ctk.CTkFrame(self)
        self.options_frame.pack(side="left", ipadx=10, ipady=10, fill="y", expand=False, padx=10, pady=10)

        self.chart_frame = ctk.CTkFrame(self)
        self.chart_frame.pack(side="right", ipadx=10, ipady=10, fill="both", expand=True, padx=10, pady=10)

        self.canvas = ctk.CTkCanvas(self.chart_frame, bg="black")
        self.canvas.pack(expand=True, fill="both")

        self.entry_initial_population = ctk.CTkEntry(self.options_frame)
        self.entry_initial_population.insert(0, "20")  # Valor inicial
        self.entry_initial_population.pack(pady=10)

        self.entry_population_size = ctk.CTkEntry(self.options_frame)
        self.entry_population_size.insert(0, "50")  # Valor inicial
        self.entry_population_size.pack(pady=10)

        self.entry_percent_of_elitism = ctk.CTkEntry(self.options_frame)
        self.entry_percent_of_elitism.insert(0, "40")  # Valor inicial
        self.entry_percent_of_elitism.pack(pady=10)

        self.entry_mutation_rate = ctk.CTkEntry(self.options_frame)
        self.entry_mutation_rate.insert(0, "70")  # Valor inicial
        self.entry_mutation_rate.pack(pady=10)

        self.entry_mutation_rate_gen = ctk.CTkEntry(self.options_frame)
        self.entry_mutation_rate_gen.insert(0, "70")  # Valor inicial
        self.entry_mutation_rate_gen.pack(pady=10)

        self.entry_breakpoint_line = ctk.CTkEntry(self.options_frame)
        self.entry_breakpoint_line.insert(0, "0")  # Valor inicial
        self.entry_breakpoint_line.pack(pady=10)

        self.entry_number_of_generations = ctk.CTkEntry(self.options_frame)
        self.entry_number_of_generations.insert(0, "30")  # Valor inicial
        self.entry_number_of_generations.pack(pady=10)

        self.start_button = ctk.CTkButton(self.options_frame, text="Start", command=lambda: self.run())
        self.start_button.pack(pady=20)

        self.members_button = ctk.CTkButton(self.options_frame, text="members video",
                                            command=lambda: self.clear_and_play_gif("members.gif"))
        self.members_button.pack(pady=10)

        self.pop_ev_button = ctk.CTkButton(self.options_frame, text="pop ev video",
                                           command=lambda: self.clear_and_play_gif("pop_ev.gif"))
        self.pop_ev_button.pack(pady=10)

        self.des_aprox_button = ctk.CTkButton(self.options_frame, text="des aprox video",
                                              command=lambda: self.clear_and_play_gif("des_aprox.gif"))
        self.des_aprox_button.pack(pady=10)

        # Parametros
        self.problem_df = pd.read_excel("data.xlsx", skiprows=1)

        self.problem_list: List[Problem] = []
        self.population: List[Member] = []

        self.initial_population = int(self.entry_initial_population.get())
        self.population_size = int(self.entry_population_size.get())
        self.percent_of_elitism = int(self.entry_percent_of_elitism.get())
        self.mutation_rate = int(self.entry_mutation_rate.get())
        self.mutation_rate_gen = int(self.entry_mutation_rate_gen.get())
        self.breakpoint_line = int(self.entry_breakpoint_line.get())
        self.number_of_generations = int(self.entry_number_of_generations.get())

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
        for member in self.population:  # iter over population
            if member["fitness_err"] == None:  # if member has not been evaluated
                member_err: List[float] = []  # create list to store error values
                for value in self.problem_list:
                    y = value["y"]  # get y value
                    y_calc = member["chromosome"][0] + np.array(
                        [member["chromosome"][x + 1] * value["x_vec"][x] for x in range(0, 4)]).sum()  # calculate y
                    member_err.append(y_calc - y)
                    member["y_calc"].append(y_calc)

                member_err_array = np.array(member_err)  # convert list to array
                member["fitness_err"] = np.linalg.norm(
                    member_err_array)  # calculate the error norm and assign to member

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
                first_children["chromosome"] = np.concatenate(
                    (best["chromosome"][:self.breakpoint_line], rest["chromosome"][self.breakpoint_line:]))
                seconde_children["chromosome"] = np.concatenate(
                    (rest["chromosome"][:self.breakpoint_line], best["chromosome"][self.breakpoint_line:]))

                # Append the new members to the population
                self.population.append(first_children)
                self.population.append(seconde_children)

        # remove the repeated members
        seen = set()
        self.population = [x for x in self.population if
                           not (tuple(x["chromosome"]) in seen or seen.add(tuple(x["chromosome"])))]

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

        pop_ev_animation = FuncAnimation(fig_pop_ev, self.population_evo_update, frames=len(self.best_members),
                                         repeat=False)
        des_aprox_animation = FuncAnimation(fig_des_aprox, self.y_des_and_aproximate_update,
                                            frames=len(self.best_members), repeat=False)
        members_animation = FuncAnimation(fig_members, self.members_update, frames=len(self.best_members), repeat=False)

        print("complete video")

        pop_ev_animation.save("pop_ev.gif", writer=PillowWriter(fps=2))
        des_aprox_animation.save("des_aprox.gif", writer=PillowWriter(fps=2))
        members_animation.save("members.gif", writer=PillowWriter(fps=2))

    def population_evo_update(self, frame):
        self.ax_pop_ev.clear()

        self.ax_pop_ev.title.set_text("Population Evolution")
        self.ax_pop_ev.set_xlabel("Generation")
        self.ax_pop_ev.set_ylabel("Fitness Error")

        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))],
                            [m["fitness_err"] for m in self.best_members[:frame]], label="Best members")
        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))], self.median_values[:frame],
                            label="Median values")
        self.ax_pop_ev.plot([i for i in range(len(self.best_members[:frame]))],
                            [m["fitness_err"] for m in self.worst_members[:frame]], label="Worst members")
        self.ax_pop_ev.legend(labels=["Best members", "mean", "Worst members"])

    def y_des_and_aproximate_update(self, frame):
        self.ax_des_apro.clear()

        self.ax_des_apro.title.set_text("Y Desired and Aproximate")
        self.ax_des_apro.set_xlabel("member")
        self.ax_des_apro.set_ylabel("Y")

        self.ax_des_apro.plot([i for i in range(len(self.y_values))], self.y_values, label="Y Desired")
        self.ax_des_apro.plot([i for i in range(len(self.y_values))], self.best_members[frame]["y_calc"],
                              label="Y Aproximate")

    def members_update(self, frame):
        self.ax_members.clear()

        self.ax_members.title.set_text("Members")
        self.ax_members.set_xlabel("member")
        self.ax_members.set_ylabel("Chromosome values")

        # A
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))],
                             [m["chromosome"][0] for m in self.best_members[:frame]], label="A")
        # B
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))],
                             [m["chromosome"][1] for m in self.best_members[:frame]], label="B")
        # C
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))],
                             [m["chromosome"][2] for m in self.best_members[:frame]], label="C")
        # D
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))],
                             [m["chromosome"][3] for m in self.best_members[:frame]], label="D")
        # E
        self.ax_members.plot([i for i in range(len(self.best_members[:frame]))],
                             [m["chromosome"][4] for m in self.best_members[:frame]], label="E")

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

    def clear_and_play_gif(self, gif_filename):
        # Clear the canvas
        self.canvas.delete("all")

        # Play the GIF
        self.play_gif(gif_filename, 0)

    def play_gif(self, gif_filename, frame):
        img = Image.open(gif_filename)
        gif_frames = []

        try:
            while True:
                gif_frames.append(ImageTk.PhotoImage(img.copy().convert('RGBA')))
                img.seek(len(gif_frames))  # Seek to the next frame
        except EOFError:
            pass  # Reached the end of the gif

        def update_frame(frame):
            if frame < len(gif_frames):
                self.canvas.create_image(0, 0, anchor="nw", image=gif_frames[frame])
                self.after(100, update_frame, frame + 1)
            else:
                self.canvas.delete("all")  # Delete the image when the GIF ends

        update_frame(frame)


if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.mainloop()
