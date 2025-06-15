import { Html, Head, Main, NextScript } from 'next/document';
import Script from 'next/script';

export default function Document() {
  return (
    <Html lang="es">
      <Head>
        {/* Aquí no se coloca nada más */}
      </Head>
      <body>
        {/* Carga asincrónica de la API de Tableau */}
        <Script
          src="https://public.tableau.com/javascripts/api/tableau-2.min.js"
          strategy="beforeInteractive"
        />
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
