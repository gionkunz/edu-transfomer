# Spec Delta

## Purpose

Define the **`llm learn`** command: an interactive lesson in the terminal that explains, step by step and with values captured from the user's own checkpoint and prompt, how a transformer predicts the next token.

## ADDED Requirements

### Requirement: Learn command invocation

The CLI SHALL provide **`llm learn [prompt...]`** with **`--model <path>`** (required), **`--tokenizer <path>`**, **`--temperature <x>`**, **`--top-k <k>`** and **`--dtype <d>`** with the same meaning and defaults as in **`llm generate`**, and **`--no-animation`**. When no prompt is given a built-in default prompt SHALL be used. The lesson SHALL explain the model's prediction of the token that follows the prompt.

#### Scenario: Lesson starts on a checkpoint

- **WHEN** the user runs **`llm learn --model <dir> "Once upon a time"`** in an interactive terminal with WebGPU available
- **THEN** the lesson SHALL open on its first step, showing the prompt split into tokens and naming the last token as the one whose next token is being predicted

#### Scenario: Prompt too long

- **WHEN** the tokenized prompt is longer than the model's maximum context minus one
- **THEN** the command SHALL exit non-zero with a message stating the prompt length in tokens and the limit

#### Scenario: Works with older checkpoints

- **WHEN** the checkpoint was written by a build that predates this command
- **THEN** the lesson SHALL run without requiring any conversion of the checkpoint or tokenizer

---

### Requirement: Learn requires an interactive terminal of sufficient size

**`llm learn`** SHALL require an interactive terminal. When the terminal is smaller than 80 columns by 24 rows the lesson SHALL replace its content with a message stating the current and required size, and SHALL restore the lesson at the same step once the terminal is large enough.

#### Scenario: Non-interactive invocation

- **WHEN** **`llm learn`** is run with standard input or output not attached to a terminal
- **THEN** the command SHALL exit non-zero with a message saying the lesson needs an interactive terminal, without loading the model

#### Scenario: Terminal resized below the minimum

- **WHEN** the terminal is resized to fewer than 80 columns or 24 rows during the lesson
- **THEN** the size message SHALL be shown, and enlarging the terminal SHALL bring back the step that was showing

---

### Requirement: Lesson structure and navigation

The lesson SHALL consist of ordered **chapters**, each made of ordered **steps**; one step is one screen. **`→`** SHALL go to the next step and **`←`** to the previous one, crossing chapter boundaries. A digit key SHALL jump to the first step of the chapter with that number. **`q`** and **`Ctrl+C`** SHALL quit and restore the terminal. Navigation SHALL never re-run the model: all steps of one lesson describe the same single prediction.

#### Scenario: Stepping across a chapter boundary

- **WHEN** the user presses **`→`** on the last step of a chapter that is not the last chapter
- **THEN** the first step of the next chapter SHALL be shown

#### Scenario: Ends of the lesson

- **WHEN** the user presses **`←`** on the first step or **`→`** on the last step
- **THEN** the step SHALL stay unchanged

#### Scenario: Values are stable while navigating

- **WHEN** the user leaves a step and returns to it without changing block, head or sampler settings
- **THEN** every number shown on that step SHALL be the same as before

---

### Requirement: Step screen layout

Every step screen SHALL show: a header with the chapter number and title, the step's position within the chapter and a progress indicator across the whole lesson; a schematic area; an explanation of at most four lines in plain language that refers to the values on screen; and a footer listing the keys that are active on that step, including what **`↑`/`↓`** currently control.

#### Scenario: Footer reflects the active controls

- **WHEN** a step has a selectable block
- **THEN** the footer SHALL state that **`↑`/`↓`** change the block, and on a step where they change the temperature it SHALL state that instead

#### Scenario: Screen fits the minimum terminal

- **WHEN** any step is shown in an 80 by 24 terminal
- **THEN** header, schematic, explanation and footer SHALL all be visible without wrapping or scrolling

---

### Requirement: Step animations

A step MAY have an animation: a fixed sequence of frames that builds up the step's schematic and ends on the complete picture. The animation SHALL start when the step is entered. **`space`** SHALL pause a running animation, resume a paused one, and replay a finished one. Leaving a step SHALL stop its animation. With **`--no-animation`** every step SHALL be shown directly in its final frame and **`space`** SHALL have no effect.

#### Scenario: Animation ends on the full picture

- **WHEN** a step's animation has finished
- **THEN** the screen SHALL be identical to that step shown with **`--no-animation`**

#### Scenario: Space replays

- **WHEN** the user presses **`space`** after an animation has finished
- **THEN** the animation SHALL run again from its first frame

#### Scenario: Navigation during an animation

- **WHEN** the user presses **`→`** while an animation is running
- **THEN** the next step SHALL be shown immediately and the previous animation SHALL not continue

---

### Requirement: Chapter — one attention head

The lesson SHALL contain a chapter that explains one attention head for the token being processed, as five steps in this order: the token's **query**; the **key** of every token so far, compared with the query; the resulting **scores**; the scores turned into **weights**; the **values** mixed by those weights into the head's output.

#### Scenario: Vectors and bars

- **WHEN** a step of this chapter is shown
- **THEN** vectors SHALL be drawn as strips of cells whose shading reflects each component, and scores and weights as labelled bars beside the token they belong to

#### Scenario: Weights follow from scores

- **WHEN** the weights step is shown
- **THEN** the token with the highest score SHALL have the largest weight, and the weights shown SHALL sum to 1 within rounding

#### Scenario: Comparison animation

- **WHEN** the keys step is animated
- **THEN** the tokens SHALL be highlighted one after another, each receiving its score as it is compared with the query

#### Scenario: Prompt longer than the screen

- **WHEN** the prompt has more tokens than the schematic has rows
- **THEN** the most recent tokens SHALL be shown, the screen SHALL say how many earlier tokens are not shown, and the weight held by the hidden tokens SHALL be stated as one total

---

### Requirement: Selecting block and head

On every step of the attention-head chapter **`↑`/`↓`** SHALL select the block and **`h`/`H`** the next or previous head within it, wrapping around. The header SHALL name the selected block and head. Changing the selection SHALL update the step with that head's values and restart its animation. The lesson SHALL start on the last block's head with the most concentrated weights.

#### Scenario: Head changes the picture

- **WHEN** the user presses **`h`** on the weights step
- **THEN** the weights of the next head of the same block SHALL be shown and the header SHALL name that head

#### Scenario: Block selection carries into the next chapter

- **WHEN** the user selects a block in the attention-head chapter and moves to the stack chapter
- **THEN** that block SHALL be the one marked as focused there

---

### Requirement: Chapter — the stack of blocks

The lesson SHALL contain a chapter that shows the whole model in data-flow order with the residual stream as a continuous line: token and position vectors entering, every block, and the output stage. Its steps SHALL introduce, in order: the stream entering the first block; what one block adds; and the model's best guess after each block. **`↑`/`↓`** SHALL move the focused block.

#### Scenario: Block contents

- **WHEN** a block is shown in this chapter
- **THEN** it SHALL show its attention update, its feed-forward update and its best guess, with the same quantities as the **`llm generate`** architecture view

#### Scenario: Guess forms through the depth

- **WHEN** the best-guess step is animated
- **THEN** blocks SHALL be revealed from first to last, each showing its most likely next token and that token's probability

#### Scenario: Last block equals the output

- **WHEN** the best-guess step is complete
- **THEN** the guess shown for the last block SHALL be the most likely token of the raw distribution shown in the sampling chapter

---

### Requirement: Chapter — picking the next token

The lesson SHALL contain a chapter that explains sampling as four steps in this order: the most likely tokens with their **raw** probabilities; the same tokens after **temperature**; the candidates kept by **top-k**; the **draw**. **`↑`/`↓`** SHALL change the temperature, **`k`** SHALL cycle top-k, and **`r`** SHALL draw again. The current temperature and top-k SHALL be shown.

#### Scenario: Top-k settings

- **WHEN** the user presses **`k`** repeatedly
- **THEN** top-k SHALL cycle through a fixed list that includes off and the value given on the command line, and candidates outside the kept set SHALL be marked as removed

#### Scenario: Lower temperature sharpens

- **WHEN** the user lowers the temperature on the temperature step
- **THEN** the probability shown for the most likely token SHALL not decrease, and the bars SHALL animate from their old to their new lengths

#### Scenario: Redraw can pick another token

- **WHEN** the user presses **`r`** repeatedly with a temperature above zero and more than one candidate kept
- **THEN** each press SHALL mark a token drawn at random from the shown final distribution, which need not be the most likely one

#### Scenario: Shown distribution is the sampler's distribution

- **WHEN** the draw step is shown with the command-line temperature and top-k
- **THEN** the final probabilities SHALL equal those **`llm generate`** would sample from for the same prompt, checkpoint, temperature and top-k with no repetition penalty

---

### Requirement: Continue with the sampled token

On the last step of the lesson **`n`** SHALL append the currently drawn token to the prompt and start the lesson again from its first step for the next position, keeping the selected block, head, temperature and top-k. The lengthened prompt SHALL be shown. When the context is full **`n`** SHALL be unavailable and the footer SHALL say so.

#### Scenario: Lesson advances one token

- **WHEN** the user presses **`n`** after the token **`" time"`** was drawn
- **THEN** the lesson SHALL restart with **`" time"`** as the last prompt token and explain the prediction of the token after it

---

### Requirement: Lesson values come from the model and inference packages

Every vector, score, weight, update size and probability shown by **`llm learn`** SHALL be computed in **`@llm-transformer-diy/model`** or **`@llm-transformer-diy/inference`** from one instrumented forward of the loaded checkpoint; **`apps/cli`** SHALL only format and draw them. Requesting the lesson capture SHALL NOT change the logits the forward produces.

#### Scenario: Weights match scores

- **WHEN** a head's captured scores and weights are compared
- **THEN** the weights SHALL equal the softmax of the scores, within floating-point tolerance

#### Scenario: Capture does not change the result

- **WHEN** the same token is decoded on the same cache state with and without the lesson capture
- **THEN** the returned logits SHALL be equal

#### Scenario: Scores match query and keys

- **WHEN** a head's captured query, keys and scores are compared
- **THEN** each score SHALL equal the dot product of the query with that token's key divided by the square root of the head dimension, within floating-point tolerance

#### Scenario: Other commands pay nothing

- **WHEN** **`llm generate`** or **`llm chat`** runs
- **THEN** no lesson capture SHALL be requested and no additional values SHALL be read back
