import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://amdresearch.github.io",
  base: "/IntelliKit",
  integrations: [
    starlight({
      title: "IntelliKit",
      logo: {
        src: "./src/assets/intellikit.svg",
      },
      social: {
        github: "https://github.com/AMDResearch/IntelliKit",
      },
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Installation", slug: "getting-started/installation" },
            { label: "Quick Start", slug: "getting-started/quickstart" },
          ],
        },
        {
          label: "Tools",
          items: [
            { label: "Kerncap", slug: "tools/kerncap" },
            { label: "Metrix", slug: "tools/metrix" },
            { label: "Linex", slug: "tools/linex" },
            { label: "Nexus", slug: "tools/nexus" },
            { label: "Accordo", slug: "tools/accordo" },
            { label: "ROCm MCP", slug: "tools/rocm-mcp" },
            { label: "uProf MCP", slug: "tools/uprof-mcp" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "MCP Setup", slug: "guides/mcp-setup" },
            { label: "End-to-End Workflow", slug: "guides/workflow" },
          ],
        },
        { label: "Contributing", slug: "contributing" },
      ],
    }),
  ],
});
