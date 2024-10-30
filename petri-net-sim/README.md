# Petri-Net Simulation

## 1. Overview

This project describes a simple software tool for simulating Petri networks, designed in Python. 

This repository is organized as follows:

```Petri-Net-Sim    : the repository root directory
            \__ include : contains the simulators backend source code.
            \__ sim : contains the simulators tool code 
            \__ gui : contains the graphical representation tool. 
            \__ docs: contains the technical report submited for this project.
            \__ examples : contains example configurations and simulations on Petri Nets.
            \__ README.md
```

In the following sections we provide an introduction to what Petri nets are, why they are important in CIM and why the need of simulation tools to analyze them. In the second section we describe how to use the PNS python package and simulation tool, how to configure the simulator, how to run it and what are the results we can get from it. Last but not least we mention how the simulator can be used to assist in the building of a Petri Net - based controller for an automated manufacturing process/ plant. 

## 2. Introduction

### 2.1 What are Petri Nets and why they are usefull
Petri networks is one of the many mathematical modeling tools used to describe and analyze systems characterized by concurrency, synchronization, and resource sharing, such as distributed systems and automated processes such as manufactoring plants [wikipedia link]. Specifically in manufacturing systems, interest in Petri nets arose from the need to define and model discrete production systems. Key characteristics of these systems include parallelism and non-determinism, which imply that actions within the systems can be executed in multiple ways. During the design of such systems, it is easy to overlook significant points of interaction, potentially leading to malfunction during operation. Therefore to manage the complexity of modern parallel systems it is essential to provide methods that enable verification and correction before deployment. One approach to address this challenge is to build an executable model of the system using simulation to provide a comprehencive view of its design and operation [thsis shit].

### 2.2 Petri Net Basics

A PN is a directed biparte graph composet by two types of elements: places and transitions. Place elements are depicted usually with a circle and transition elements are depicted usually with a rectangle. Places communicate with transitions with directed arcs. Arcs never run in between places or between transitions. A place from which an arc runs to a transition are called input places of the transition, while the places to chich arcs run from a transition are called the output places of the transition. 

Places in a PN may contain a discrete number of marks called tokens. Any distribution of tokens over the places will represent a configuration of the net called a marking. A transition in a PN can fire if it is enabled. For the transition to fire there has to be sufficient tokens in all of its inputs as defined by the weights assigned to the arcs that connect the input places to the transition. When a transition fires it consumes the required input tokens and creates tokens in its output places accordingly based on the weight of the arc that connects the transition with the output place. Every fire of transitions is atomic, which means that it is a single non-interruptable step.  

### 2.3 Extensions of Petri Nets / Need for Simulators

One of such extensions is the inhibitory Petri nets. An inhibitor Petri net uses an inhibitor arc which imposes the precondition that the transition may only fire when the place is empty. Unlike regular arcs that enable transitions when tokens are present, an inhibitory arc blocks the firing of the transition when tokens exist in the place its linked to.

This feature allows for more precise control over transitions, creating conditions where certain transitions can only fire if certain places are empty. This is particularly useful for representing situations where an action should only occur when a specific resource or condition is absent.

Inhibitory nets can model behaviors that standard Petri nets cannot, such as complex dependencies, priority control, and specific concurrency rules. They are commonly applied in fields like manufacturing systems, computer science, and control systems, where specific conditions or resources must be absent for an action to proceed.

The classical Petri nets can be analyzed formally only through mathematical modeling. As the size of the PN increases solving the mathematical operations that are required become very demanding in means of computational power. Also the new enhanced Petri nets such as the one that support inhibitory arcs as the one described here cannot be solved formally but rather only through a simulation progrma. Thats the reason for which a simulator is needed to solve the Petri nets.

### 2.4 Simulation Capabilities

A PN simulator offers a wide range of capabilities to systems that are described through the use of Petri nets. 

## Setting Up the Simulator

### Configuration File

### Termination Condition

### Run Command

## Results