---
version: alpha
name: Sleeper
description: >-
  Real-time fantasy sports platform delivering live scores, standings, and player updates across NFL, NBA, CBB, MLB,
  NHL, and WNBA with a high-velocity, data-dense interface.
logo:
  src: https://sleepercdn.com/landing/web2026/img/logos/logo-full-horizontal-white.png
  srcDark: https://sleepercdn.com/landing/web2026/img/logos/logo-full-horizontal-white.png
colors:
  surface: '#050921'
  surface-dim: '#030614'
  surface-bright: '#0a0f2a'
  surface-container-lowest: '#020409'
  surface-container-low: '#0a0f2a'
  surface-container: '#131b38'
  surface-container-high: '#1a2447'
  surface-container-highest: '#242d52'
  on-surface: '#ffffff'
  on-surface-variant: '#d8d8d8'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#050921'
  outline: '#696969'
  outline-variant: '#343855'
  surface-tint: '#00fff9'
  primary: '#00fff9'
  on-primary: '#050921'
  primary-container: '#00ceb8'
  on-primary-container: '#050921'
  inverse-primary: '#00baff'
  secondary: '#3860be'
  on-secondary: '#ffffff'
  secondary-container: '#4c5e93'
  on-secondary-container: '#e2e2e2'
  tertiary: '#ffae58'
  on-tertiary: '#050921'
  tertiary-container: '#ff6f42'
  on-tertiary-container: '#ffffff'
  error: '#ff0000'
  on-error: '#ffffff'
  error-container: '#ff6b6b'
  on-error-container: '#050921'
  primary-fixed: '#00fff9'
  primary-fixed-dim: '#00d7ff'
  on-primary-fixed: '#050921'
  on-primary-fixed-variant: '#131b38'
  secondary-fixed: '#3860be'
  secondary-fixed-dim: '#2a4a8f'
  on-secondary-fixed: '#ffffff'
  on-secondary-fixed-variant: '#e2e2e2'
  tertiary-fixed: '#ffae58'
  tertiary-fixed-dim: '#ff9940'
  on-tertiary-fixed: '#050921'
  on-tertiary-fixed-variant: '#131b38'
  background: '#050921'
  on-background: '#ffffff'
  surface-variant: '#1a2447'
typography:
  display:
    fontFamily: Poppins
    fontSize: 60px
    fontWeight: '700'
    lineHeight: 68px
    letterSpacing: '-0.04em'
  headline-lg:
    fontFamily: Poppins
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: '-0.02em'
  headline-md:
    fontFamily: Poppins
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: '-0.01em'
  title-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: 0em
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: 0em
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  label-md:
    fontFamily: Poppins
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Poppins
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 2px
  DEFAULT: 4px
  md: 10px
  lg: 16px
  xl: 20px
  full: 9999px
spacing:
  unit: 8px
  xs: 4px
  sm: 12px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  container-max: 1440px
elevation:
  sm: 0 1px 2px rgba(0, 0, 0, 0.12)
  md: 0 3px 8px rgba(0, 0, 0, 0.15)
  lg: 0 8px 24px rgba(0, 0, 0, 0.2)
layout:
  containerMaxWidth: 1440px
  gridColumns: 12
components:
  button-primary:
    backgroundColor: '{colors.primary}'
    textColor: '{colors.on-primary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 12px 24px
    height: 44px
    border: none
  button-primary-hover:
    backgroundColor: '{colors.primary-fixed-dim}'
    textColor: '{colors.on-primary}'
    transition: background-color 150ms ease-out
  button-secondary:
    backgroundColor: '{colors.secondary}'
    textColor: '{colors.on-secondary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.full}'
    padding: 8px 16px
    height: 36px
    border: none
  button-secondary-hover:
    backgroundColor: '{colors.secondary-fixed-dim}'
    transition: background-color 150ms ease-out
  button-tertiary:
    backgroundColor: transparent
    textColor: '{colors.primary}'
    typography: '{typography.label-md}'
    rounded: '{rounded.DEFAULT}'
    padding: '{spacing.sm}'
    border: 1px solid {colors.outline-variant}
  button-tertiary-hover:
    backgroundColor: rgba(0, 255, 249, 0.08)
    borderColor: '{colors.primary}'
  card:
    backgroundColor: rgba(19, 27, 56, 0.3)
    rounded: '{rounded.md}'
    padding: '{spacing.md}'
    border: 1px solid rgba(52, 56, 85, 0.4)
    backdropFilter: blur(8px)
  card-hover:
    backgroundColor: rgba(26, 36, 71, 0.4)
    borderColor: rgba(0, 255, 249, 0.2)
    transition: all 200ms ease-out
  input-field:
    backgroundColor: rgba(0, 0, 0, 0.2)
    textColor: '{colors.on-surface}'
    typography: '{typography.body-md}'
    rounded: '{rounded.DEFAULT}'
    padding: 20px 16px
    border: 1px solid rgba(0, 255, 249, 0.3)
    height: 56px
  input-field-focus:
    borderColor: '{colors.primary}'
    boxShadow: 0 0 0 3px rgba(0, 255, 249, 0.1)
    transition: border-color 150ms ease-out, box-shadow 150ms ease-out
  modal-overlay:
    backgroundColor: rgba(5, 9, 29, 0.8)
    backdropFilter: blur(4px)
  modal-content:
    backgroundColor: '{colors.surface-container}'
    rounded: '{rounded.lg}'
    padding: '{spacing.lg}'
    border: 1px solid rgba(0, 255, 249, 0.15)
    boxShadow: 0 8px 32px rgba(0, 0, 0, 0.3)
  badge-success:
    backgroundColor: '#0a5d21'
    textColor: '#28e757'
    typography: '{typography.label-sm}'
    rounded: '{rounded.full}'
    padding: 4px 12px
  badge-accent:
    backgroundColor: rgba(0, 255, 249, 0.15)
    textColor: '{colors.primary}'
    typography: '{typography.label-sm}'
    rounded: '{rounded.full}'
    padding: 4px 12px
  list-item:
    backgroundColor: transparent
    rounded: '{rounded.md}'
    padding: '{spacing.sm}'
    border: none
  list-item-hover:
    backgroundColor: rgba(0, 255, 249, 0.06)
    transition: background-color 120ms ease-out
---

## Overview

Sleeper is a high-velocity fantasy sports platform engineered for real-time data consumption and rapid decision-making. The design system embodies 'Velocity Minimalism'—a aesthetic that prioritizes information density, instant visual feedback, and competitive urgency while maintaining clarity under cognitive load. The interface uses a deep navy-to-black canvas (rgb(5, 9, 29)) as a neutral stage, allowing vibrant accent colors (cyan #00fff9, electric blue #00baff, warm orange #ffae58) to command attention without fatigue. The brand personality is direct, data-driven, and uncompromising: users expect millisecond-precision updates, zero visual clutter, and immediate access to league standings and player scores. Voice: terse, action-oriented, never apologetic. Example: 'Your league updated 47ms ago—check the matchup.'

## Colors

The color system is built on a dark-first philosophy optimized for extended viewing and real-time monitoring. Primary (rgb(0, 255, 249) / #00fff9) is the brand's signature accent—deployed on all CTAs ('SIGN UP', 'CREATE AN ACCOUNT'), active navigation states, and input focus rings. It commands 3–5% of screen real estate but drives 80% of interaction cues. Secondary (rgb(56, 96, 190) / #3860be) anchors secondary actions and data-layer highlights (e.g., league badges, status indicators). Tertiary (rgb(255, 174, 88) / #ffae58) is reserved for alerts, warnings, and time-sensitive notifications. The surface stack descends from #050921 (surface) through #131b38 (surface-container, used for modals and elevated cards) to #020409 (surface-container-lowest, for deeply nested UI). All surfaces use rg

## Typography

The type system pairs Poppins (display, headlines, labels) for high-contrast, geometric clarity with Inter (body, data-dense tables) for neutral readability at small scales. Display (60px, 700 weight, -0.04em tracking) is reserved for hero moments and league names. Headline-lg (40px, 600 weight) anchors section headers. Body-md (16px, 400 weight, 24px line-height) is the workhorse for score tables and player stats—the 24px line-height (1.5x) prevents crowding in data-heavy layouts. Label-md (14px, 600 weight, 0.01em tracking) is applied to all interactive elements (buttons, badges, tabs) and must include a 150ms transition on hover to signal interactivity. On small labels over busy backgrounds (e.g., player names over team logos), apply text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) to ensure 7

## Layout

The layout uses a 12-column fluid grid with a 1440px container max-width, collapsing to 100% viewport width on mobile. Horizontal rhythm is governed by the 24px gutter (md spacing token), creating breathing room between league cards and score tiles without sacrificing density. Vertical rhythm uses lg spacing (40px) to separate major sections (header, matchup grid, standings table), md spacing (24px) for card-to-card separation, and sm spacing (12px) for intra-card element spacing. The header is fixed at the top with a semi-transparent backdrop (rgba(5, 9, 29, 0.8) + blur(4px)) to maintain navigation access while scrolling through live scores. Modals use a centered overlay with surface-container background and 32px padding, constrained to 600px max-width on desktop. All spacing increments a

## Elevation & Depth

Depth is conveyed through layered backdrop-filter effects and subtle shadows rather than flat color shifts. Level 1 (Base): the background gradient (dark navy to near-black) with no shadow. Level 2 (Standard Card): rgba(19, 27, 56, 0.3) background + backdrop-filter: blur(8px) + 1px border at rgba(52, 56, 85, 0.4) + box-shadow: 0 3px 8px rgba(0, 0, 0, 0.15). Level 3 (Modal/Elevated): rgba(26, 36, 71, 0.4) background + backdrop-filter: blur(12px) + 1px border at rgba(0, 255, 249, 0.15) + box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3). Hover states transition the shadow to 0 6px 16px rgba(0, 0, 0, 0.2

## Shapes

The shape philosophy is 'Technical Precision'—sharp, geometric, and purposeful. Buttons use full (9999px) border-radius to signal primary actions and create a pill-shaped affordance that feels modern and clickable. Secondary buttons and tertiary actions use DEFAULT (4px) for a more restrained, data-table aesthetic. Cards and modals use md (10px) for a balance between approachability and technical rigor. Input fields use DEFAULT (4px) to align with form conventions and maintain visual consistency with data tables. The rationale: full-radius buttons stand out in a sea of sharp UI, making CTAs un

## Components

### Action Elements
Buttons are the primary interaction primitive. Button-primary (bg: #00fff9, text: #050921, 44px height, 12px 24px padding, full radius) uses Poppins 600 at 14px and must include a 150ms ease-out transition on hover to darken to #00d7ff. Button-secondary (bg: #3860be, text: white, 36px height, 8px 16px padding) is used for secondary CTAs like 'Log In' and 'Create Account'. Button-tertiary (transparent bg, cyan text, 1px outline) is for low-priority actions and must show a 0.06 opacity cyan background on hover. All buttons disable at 50% opacity with cursor: not-allowed.

### Containers & Surfaces
Cards use rgba(19, 27, 56, 0.3) background with backdrop-filter: blur(8px), 1px border at rgba(52, 56, 85, 0.4), and 24px padding. On hover, the background shifts to rgba(26, 36

## Do's and Don'ts

**Do**
- Do use cyan (#00fff9) for all primary CTAs and focus states—it is the brand's signature and must be instantly recognizable.
- Do maintain 24px gutters between major sections and 12px between card elements to prevent visual crowding in data-dense layouts.
- Do apply 150ms ease-out transitions to all interactive elements (buttons, cards, inputs) to signal state changes without lag.
- Do use Poppins 600 for all labels and headlines to create visual hierarchy; reserve Inter 400 for body text and data tables.
- Do include backdrop-filter: blur(8px) on all elevated surfaces (cards, modals) to create depth and maintain the glassmorphic aesthetic.
- Do test all color combinations against the dark background for 4.5:1 WCAG AA contrast, especially on small text.

**Don't**
- Don't use border-radius values between 5px and 9px—stick to the defined scale (sm: 2px, DEFAULT: 4px, md: 10px, lg: 16px, xl: 20px, full: 9999px).
- Don't apply shadows without a corresponding backdrop-filter blur; the system relies on layered transparency, not depth alone.
- Don't exceed 5% of screen real estate with primary accent color (#00fff9)—overuse dilutes urgency and causes visual fatigue.
- Don't use system fonts or generic sans-serif fallbacks; always specify Poppins or Inter with proper @font-face declarations and fallback stacks.
