// Main JavaScript file for DermaTRACE
document.addEventListener("DOMContentLoaded", () => {
    // Add smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');

    for (const link of links) {
        link.addEventListener("click", function (e) {
            const href = this.getAttribute("href");

            if (href !== "#") {
                e.preventDefault();
                const target = document.querySelector(href);

                if (target) {
                    target.scrollIntoView({
                        behavior: "smooth",
                    });
                }
            }
        });
    }

    // Add animation to cards when they come into view
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("animate-in");
                }
            });
        },
        { threshold: 0.1 }
    );

    document.querySelectorAll(".info-card, .action-btn, .stat-card, .step").forEach((element) => {
        observer.observe(element);
    });

    // Dark mode toggle
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');

            // Save preference to localStorage
            if (document.body.classList.contains('dark-mode')) {
                localStorage.setItem('theme', 'dark');
            } else {
                localStorage.setItem('theme', 'light');
            }
        });

        // Check for saved theme preference
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-mode');
        }
    }

    // Testimonial slider
    const testimonials = document.querySelectorAll('.testimonial');
    const dots = document.querySelectorAll('.dot');

    if (testimonials.length > 0 && dots.length > 0) {
        let currentTestimonial = 0;

        // Function to show testimonial
        const showTestimonial = (index) => {
            testimonials.forEach(testimonial => testimonial.style.opacity = 0);
            dots.forEach(dot => dot.classList.remove('active'));

            testimonials[index].style.opacity = 1;
            dots[index].classList.add('active');
        };

        // Click event for dots
        dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                currentTestimonial = index;
                showTestimonial(currentTestimonial);
            });
        });

        // Auto rotate testimonials
        setInterval(() => {
            currentTestimonial = (currentTestimonial + 1) % testimonials.length;
            showTestimonial(currentTestimonial);
        }, 5000);
    }
});