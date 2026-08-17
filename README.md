# lumi-fit

Let:

- $m$: model
- $g_m$: number of billed accelerator units
- $T_m$: inference duration in hours
- $N_{in,m}$, $N_{out,m}$: total input and output tokens
- $d=0.6$: billing factor

At the selected concurrent operating point, define throughput as:

$$
\theta_{in,m}
=
\frac{N_{in,m}}{10^6 g_mT_m}
$$

$$
\theta_{out,m}
=
\frac{N_{out,m}}{10^6 g_mT_m}
$$

Both are measured in million tokens per accelerator-hour.

The model-specific Aitta BU rates are:

$$
BU_{in,m}=\frac{d}{\theta_{in,m}}
$$

$$
BU_{out,m}=\frac{d}{\theta_{out,m}}
$$

For an inference request containing \(n_{in}\) input tokens and \(n_{out}\) output tokens:

$$
BU_{\mathrm{request}}
=
\frac{n_{in}}{10^6}BU_{in,m}
+
\frac{n_{out}}{10^6}BU_{out,m}
$$

Equivalently:

$$
BU_{\mathrm{request}}
=
\frac{d}{10^6}
\left(
\frac{n_{in}}{\theta_{in,m}}
+
\frac{n_{out}}{\theta_{out,m}}
\right)
$$

Under the fixed assumption of one million input and one million output tokens:

$$
BU_m^{1:1}
=
BU_{in,m}+BU_{out,m}
=
d\left(
\frac{1}{\theta_{in,m}}+
\frac{1}{\theta_{out,m}}
\right)
$$
