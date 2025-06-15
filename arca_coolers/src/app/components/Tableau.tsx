"use client";

import { useEffect, useRef } from "react";

export default function TableauDashboard() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const divElement = document.getElementById("viz1749971859015");
    const vizElement = divElement?.getElementsByTagName("object")[0];

    if (divElement && vizElement) {
      if (divElement.offsetWidth > 800) {
        vizElement.style.width = "1366px";
        vizElement.style.height = "795px";
      } else if (divElement.offsetWidth > 500) {
        vizElement.style.width = "1366px";
        vizElement.style.height = "795px";
      } else {
        vizElement.style.width = "100%";
        vizElement.style.height = "1727px";
      }

      const scriptElement = document.createElement("script");
      scriptElement.src = "https://public.tableau.com/javascripts/api/viz_v1.js";
      vizElement.parentNode?.insertBefore(scriptElement, vizElement);
    }
  }, []);

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-center text-gray-800">
        Monitoreo Inteligente de Smart Coolers: Diagnóstico de Fallas y Desempeño Operativo
      </h2>

      <div
        className="tableauPlaceholder"
        id="viz1749971859015"
        ref={containerRef}
        style={{ position: "relative" }}
      >
        <noscript>
          <a href="#">
            <img
              alt="Dashboard 1"
              src="https://public.tableau.com/static/images/AR/ARCA_RETO/Dashboard1/1_rss.png"
              style={{ border: "none" }}
            />
          </a>
        </noscript>

        <object className="tableauViz" style={{ display: "none" }}>
          <param name="host_url" value="https%3A%2F%2Fpublic.tableau.com%2F" />
          <param name="embed_code_version" value="3" />
          <param name="site_root" value="" />
          <param name="name" value="ARCA_RETO&#47;Dashboard1" />
          <param name="tabs" value="no" />
          <param name="toolbar" value="yes" />
          <param name="static_image" value="https://public.tableau.com/static/images/AR/ARCA_RETO/Dashboard1/1.png" />
          <param name="animate_transition" value="yes" />
          <param name="display_static_image" value="yes" />
          <param name="display_spinner" value="yes" />
          <param name="display_overlay" value="yes" />
          <param name="display_count" value="yes" />
          <param name="language" value="es-ES" />
        </object>
      </div>
    </div>
  );
}
