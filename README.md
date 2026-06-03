# TruthGuard AI: Fake News Detection System

TruthGuard AI is a machine learning-based web application designed to detect whether a news article or headline is likely to be real, fake, or uncertain. The project uses Natural Language Processing (NLP) and a supervised machine learning model trained on a real fake-news dataset.

## Project Goal

The goal of this project is to demonstrate how machine learning can be applied to a practical real-world problem: fake news detection.

The system allows users to enter a news headline, paragraph, or article. After processing the text, the application predicts whether the input is more similar to real news or fake news based on patterns learned from the training dataset.

## Problem Statement

Fake news can spread misinformation, influence public opinion, and reduce trust in media sources. Manual fact-checking takes time, so machine learning can be used as an assistive tool to identify potentially unreliable content.

This project solves a binary text classification problem:

- Input: news title or article text
- Output: Real News, Fake News, or Uncertain

## Dataset

The project uses the WELFake dataset.

The dataset contains labeled news articles with the following main columns:

- `title`: title of the news article
- `text`: full article text
- `label`: classification label

Label meaning:

```text
0 = Real News
1 = Fake News