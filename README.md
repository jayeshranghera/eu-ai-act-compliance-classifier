## Project Overview
A tool that classifies AI systems according to 
EU AI Act risk levels and provides compliance 
recommendations.

## Problem
Every company deploying AI in Europe must comply 
with the EU AI Act by 2026. But the Act is 100+ 
pages of complex legal language. Small companies, 
startups, and non-EU firms (especially Indian IT 
companies) have no simple way to check if their 
AI systems are compliant.

This creates a compliance gap — organizations 
either hire expensive consultants or remain 
unaware of their obligations.

## Solution
An open-source compliance classifier where any 
organization can input their AI system description. 
The tool checks it against EU AI Act Annex I and 
Annex III, identifies the risk level (Prohibited / 
High Risk / Limited Risk / Minimal Risk), explains 
the reason, and provides actionable compliance 
recommendations.

Target users: Indian IT companies, EU startups, 
small firms expanding to European markets.

## Dataset
- EU AI Act Full Text (Official EU source — 
  ec.europa.eu)
- Annex I — Product & Safety Regulation Sectors
- Annex III — High Risk AI Use Cases
- Article 52 — Transparency Obligations
- OECD AI Incident Database (real-world AI 
  failure cases)
- Manually curated dataset of 50-100 real AI 
  system descriptions with classifications

## Tech Stack
- Python
- LangChain + RAG Pipeline
- ChromaDB (vector database)
- Streamlit (dashboard)
- HuggingFace / OpenAI (LLM)
- Docker (deployment)

## Policy Brief
This project will produce a policy brief covering:

- Classification analysis of 50-100 real AI systems
- Industry-wise compliance gap analysis
- Key findings on non-compliant AI use cases
- Recommendations for:
  - Indian IT companies expanding to EU markets
  - EU startups and small firms
  - Policymakers developing AI governance frameworks

Target audience: Think tanks, NGOs, policy 
researchers, and compliance teams.
