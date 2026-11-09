#pragma once

struct SimConfig {
    int gridW = 256;
    int gridH = 256;
};

// Velocity: grid cells / second; y points down. Dissipation is per simulation step.
struct SimParams {
    int gridW = 256; // Read-only in practice: step validates against SimConfig.
    int gridH = 256;
    int jacobiIter = 60;
    int diffuseIter = 20;
    float velDissip = 0.99f;
    float denDissip = 0.995f;
    float viscosity = 0.0f; // cells^2 / second
    float forceRadius = 0.08f; // Fraction of min(gridW, gridH).
    float forceStrength = 400.f; // Mouse acceleration multiplier.
    float buoyancy = 0.0f; // Positive density accelerates upward.
    float dt = 1.0f / 60.0f;
};

struct FluidInjection {
    float x = 0, y = 0; // Cell coordinates, origin at top left.
    float vx = 0, vy = 0; // Mouse velocity in cells / second.
    float density = 300.0f; // Density / second at brush center.
};
