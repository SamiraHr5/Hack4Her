"use client";

import { useEffect, useRef } from "react";

export default function TableauDashboard() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://public.tableau.com/javascripts/api/viz_v1.js";
    script.type = "text/javascript";
    script.async = true;

    if (containerRef.current) {
      containerRef.current.appendChild(script);
    }
  }, []);

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-center text-gray-800">
        Nuevo León Bajo la Lupa: Análisis de Corrupción en el Tránsito
      </h2>

      <div
        ref={containerRef}
        className="tableauPlaceholder"
        id="viz1749931856620"
        style={{
          position: "relative",
          width: "100%",
        }}
      >
        <noscript>
          <a href="#">
            <img
              alt="Nuevo León Bajo la Lupa"
              src="https://public.tableau.com/static/images/N5/N5KZFFW2M/1_rss.png"
              style={{ border: "none" }}
            />
          </a>
        </noscript>

        <object
          className="tableauViz"
          style={{
            width: "100%",
            height: "1100px",
            display: "none",
          }}
        >
          <param name="host_url" value="https%3A%2F%2Fpublic.tableau.com%2F" />
          <param name="embed_code_version" value="3" />
          <param name="path" value="shared/N5KZFFW2M" />
          <param name="toolbar" value="yes" />
          <param name="static_image" value="https://public.tableau.com/static/images/N5/N5KZFFW2M/1.png" />
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
