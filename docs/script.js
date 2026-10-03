// Modal functionality for the gallery
function openModal(imageSrc, captionText) {
    const modal = document.getElementById("image-modal");
    const modalImg = document.getElementById("modal-img");
    const caption = document.getElementById("caption");
    
    modal.style.display = "block";
    modalImg.src = imageSrc;
    caption.innerHTML = captionText;
    
    // Prevent scrolling on body
    document.body.style.overflow = "hidden";
}

function closeModal() {
    const modal = document.getElementById("image-modal");
    modal.style.display = "none";
    
    // Restore scrolling
    document.body.style.overflow = "auto";
}

// Close modal when clicking outside the image
document.getElementById('image-modal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === "Escape") {
        closeModal();
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        
        const targetId = this.getAttribute('href');
        if(targetId === '#') return;
        
        const targetElement = document.querySelector(targetId);
        
        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 70, // Adjust for fixed navbar
                behavior: 'smooth'
            });
        }
    });
});

// Reveal elements on scroll
function reveal() {
    var reveals = document.querySelectorAll(".card, .gallery-item");

    for (var i = 0; i < reveals.length; i++) {
        var windowHeight = window.innerHeight;
        var elementTop = reveals[i].getBoundingClientRect().top;
        var elementVisible = 100;

        if (elementTop < windowHeight - elementVisible) {
            reveals[i].style.opacity = "1";
            reveals[i].style.transform = "translateY(0)";
            reveals[i].style.transition = "all 0.6s ease";
        }
    }
}

// Set initial state for reveal
document.addEventListener("DOMContentLoaded", function() {
    var reveals = document.querySelectorAll(".card, .gallery-item");
    for (var i = 0; i < reveals.length; i++) {
        // Only apply to elements not in the hero section
        if(!reveals[i].closest('.hero')) {
            reveals[i].style.opacity = "0";
            reveals[i].style.transform = "translateY(30px)";
        }
    }
    reveal(); // Check on initial load
});

window.addEventListener("scroll", reveal);

// Function to handle leaderboard view switching
function updateLeaderboard() {
    const selector = document.getElementById('model-select');
    const selectedValue = selector.value;
    
    const overallTable = document.getElementById('overall-table');
    const mistralTable = document.getElementById('mistral-table');
    const orcaTable = document.getElementById('orca-table');
    const blankTable = document.getElementById('blank-table');
    const tables = {
        'overall': overallTable,
        'mistral': mistralTable,
        'orca': orcaTable,
        'gemma': document.getElementById('gemma-table'),
        'gpt': document.getElementById('gpt-table'),
        'qwen1b': document.getElementById('qwen1b-table'),
        'qwen3b': document.getElementById('qwen3b-table'),
        'deepseek': document.getElementById('deepseek-table'),
        'llama': document.getElementById('llama-table'),
        'gemini': document.getElementById('gemini-table')
    };
    
    // Hide all
    Object.values(tables).forEach(t => { if(t) t.style.display = 'none'; });
    blankTable.style.display = 'none';
    
    // Show selected
    if (tables[selectedValue]) {
        tables[selectedValue].style.display = 'table';
    } else {
        blankTable.style.display = 'table';
    }
}

function updateVisualizations() {
    const selector = document.getElementById('viz-model-select');
    const selectedValue = selector.value;
    
    const mistralGallery = document.getElementById('mistral-gallery');
    const orcaGallery = document.getElementById('orca-gallery');
    
    mistralGallery.style.display = 'none';
    orcaGallery.style.display = 'none';
    
    if (selectedValue === 'mistral') {
        mistralGallery.style.display = 'flex'; // Wait, what is the default display for gallery?
    } else if (selectedValue === 'orca') {
        orcaGallery.style.display = 'flex';
    }
}
